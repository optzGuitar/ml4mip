import pytorch_lightning as pl
import monai.networks.nets as nets
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torchio as tio
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassF1Score
from monai.metrics import compute_hausdorff_distance

from segmentation.config import SegmentationConfig
from segmentation.loss import CustomLoss


class SegModule(pl.LightningModule):
    def __init__(self, segmentation_config: SegmentationConfig) -> None:
        super().__init__()

        self.unet = nets.Unet(
            spatial_dims=segmentation_config.data_config.n_dims,
            in_channels=segmentation_config.data_config.n_channels,
            out_channels=segmentation_config.data_config.n_classes,
            channels=segmentation_config.model_config.channels,
            strides=segmentation_config.model_config.strides,
            dropout=segmentation_config.model_config.dropout,
        )

        self.config = segmentation_config

        self.loss = CustomLoss(
            segmentation_config,
        )

    def forward(self, subject: dict):
        sampler = tio.inference.GridSampler(
            subject, patch_size=self.hparams['patch_size'], patch_overlap=4)
        aggregator = tio.inference.GridAggregator(sampler)
        patch_loader = DataLoader(
            sampler, batch_size=self.hparams['batch_size'])
        for patch in patch_loader:
            locations = patch[tio.LOCATION]
            x, y = self.split_batch(patch)
            patch_prediction = self.unet(x)
            aggregator.add_batch(patch_prediction, locations)

        prediction = aggregator.get_output_tensor()
        return prediction

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.unet.parameters(), lr=self.config.loss_config.start_lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, self.config.loss_config.cosine_period,
                                                                            eta_min=self.config.loss_config.min_lr)
        self.unet = torch.jit.script(self.unet)
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            }
        }

    def split_batch(self, batch):
        images = torch.concat(
            [i[tio.DATA] for k, i in batch.items() if k != tio.LABEL], dim=1)
        segmentation = batch[tio.LABEL][tio.DATA]

        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            segmentation = segmentation.unsqueeze(0)

        segmentation = F.one_hot(
            segmentation.squeeze(1).to(torch.long), self.config.data_config.n_classes).permute(0, 4, 1, 2, 3).to(torch.float)

        return images, segmentation

    def metrics(self, prediction, y, is_train: bool):
        loss = self.loss(prediction, y)

        prefix = "train" if is_train else "val"

        self.log(f"{prefix}/loss", loss.mean().item())

        if is_train:
            return {
                "loss": loss.mean().item(),
            }

        dice = MulticlassF1Score(
            num_classes=self.config.data_config.n_classes).to(self.device)

        segmentations_hat_max = torch.argmax(prediction, dim=1)
        segmentation_max = torch.argmax(y, dim=1)

        dice_scores = {}
        for i, name in enumerate(['necrotic', 'edematous', 'enahncing']):
            dice_score = dice(
                (segmentations_hat_max == i+1).flatten(),
                (segmentation_max == i+1).flatten(),
            )
            dice_scores[f"dice_{name}"] = dice_score.mean().item()

            self.log(f"{prefix}/dice_{name}", dice_score.mean().item())

        seg_hat_onehot = F.one_hot(segmentations_hat_max,
                                   num_classes=self.config.data_config.n_classes)
        seg_onehot = F.one_hot(
            segmentation_max, num_classes=self.config.data_config.n_classes)
        tumor_score = compute_hausdorff_distance(
            seg_hat_onehot.swapaxes(1, -1)[:, [0, 1, 3]],
            seg_onehot.swapaxes(1, -1)[:, [0, 1, 3]],
        )
        tumor_score = tumor_score.flatten()
        tumor_score = tumor_score[~(tumor_score.isnan() | tumor_score.isinf())]

        whole_tumor = compute_hausdorff_distance(
            seg_hat_onehot.swapaxes(1, -1),
            seg_onehot.swapaxes(1, -1),
        )
        whole_tumor = whole_tumor.flatten()
        whole_tumor = whole_tumor[~(whole_tumor.isnan() | whole_tumor.isinf())]

        tumor_mean = tumor_score.mean().item()
        whole_mean = whole_tumor.mean().item()

        tumor_mean = 0 if tumor_mean is None or tumor_mean == float(
            'inf') or tumor_mean == float(
            '-inf') else tumor_mean
        whole_mean = 0 if whole_mean is None or whole_mean == float(
            'inf') or tumor_mean == float(
            '-inf') else whole_mean

        self.log("{prefix}/tumor_score", tumor_mean)
        self.log("{prefix}/whole_tumor", whole_mean)
        self.log("overall_mean", torch.as_tensor([
            *dice_scores.values(),
            tumor_mean,
            whole_mean,
        ]).mean())

        torch.cuda.empty_cache()

        return {
            "loss": loss,
            **dice_scores,
            "tumor_score": tumor_mean,
            "whole_tumor": whole_mean,
            "overall_mean": torch.as_tensor([
                *dice_scores.values(),
                tumor_mean,
                whole_mean,
            ]).mean()
        }

    def training_step(self, batch, batch_idx):
        x, y = self.split_batch(batch)
        x = x.float()

        prediction = self.unet(x)
        metrics = self.metrics(prediction, y, is_train=True)

        return metrics

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = self.split_batch(batch)
            x = x.float()

            prediction = self.unet(x)
            metrics = self.metrics(prediction, y, is_train=False)

            return metrics
