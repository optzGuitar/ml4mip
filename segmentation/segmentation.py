from typing import Any, Optional
import pytorch_lightning as pl
import monai.networks.nets as nets
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch

from segmentation.config import SegmentationConfig
import torch.nn as nn

from segmentation.loss import CustomLoss
from torchmetrics.classification import MulticlassF1Score
from monai.metrics import compute_hausdorff_distance
from torch.autograd import grad
import torch.nn.functional as F
import torch.nn as nn


class SegmentationModule(pl.LightningModule):
    def __init__(self, segmentation_config: SegmentationConfig):
        super().__init__()
        self.model = nn.Sequential(nets.SegResNet(
            spatial_dims=segmentation_config.data_config.n_dims,
            in_channels=segmentation_config.data_config.n_channels,
            out_channels=segmentation_config.data_config.n_classes,
            # img_size=segmentation_config.data_config.image_size[1:],
            # channels=segmentation_config.model_config.channels,
            # strides=segmentation_config.model_config.strides,
            # num_res_units=2,
            # kernel_size=segmentation_config.model_config.kernels,
            # up_kernel_size=list(
            #     reversed(segmentation_config.model_config.kernels)),
            dropout_prob=segmentation_config.model_config.dropout,
        ),
            nn.Softmax(dim=1)
        )

        self.loss = CustomLoss(
            segmentation_config,
        )
        self.overlap_loss = nn.MSELoss()

        self.config = segmentation_config

        self.save_hyperparameters('segmentation_config')

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        lr_scheduler = self.lr_schedulers()

        images, segmentation = self._get_from_batch(batch)
        loss, _ = self._infere(images, segmentation)
        self.log("train/lr", lr_scheduler.get_last_lr()[0])

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        images, segmentation = self._get_from_batch(batch)

        _, segmentation_hat = self._infere(
            images, segmentation, is_train=False)
        # TODO: add image logging!

        dice = MulticlassF1Score(
            average=None, num_classes=self.config.data_config.n_classes).to(self.device)

        segmentations_hat_max = torch.argmax(segmentation_hat, dim=1)
        segmentation_max = torch.argmax(segmentation, dim=1)
        for i, name in enumerate(['necrotic', 'edematous', 'enahncing']):
            dice_score = dice(
                (segmentations_hat_max == i + 1),
                (segmentation_max == i+1)
            )

            self.log(f"val/dice_{name}", dice_score.mean().item())

        tumor_score = compute_hausdorff_distance(
            F.one_hot(segmentations_hat_max,
                      num_classes=self.config.data_config.n_classes)[:, [1, 3]],
            F.one_hot(segmentation_max, num_classes=self.config.data_config.n_classes)[
                :, [1, 4]],
            include_background=True,
        )
        whole_tumor = compute_hausdorff_distance(
            F.one_hot(segmentations_hat_max,
                      num_classes=self.config.data_config.n_classes)[:, [1, 2, 3]],
            F.one_hot(segmentation_max, num_classes=self.config.data_config.n_classes)[
                :, [1, 2, 4]],
        )

        self.log("val/tumor_score", tumor_score.mean().item())
        self.log("val/whole_tumor", whole_tumor.mean().item())

        return (dice_score.mean() + tumor_score.mean() + whole_tumor.mean()).mean()

    def flatten_index_to_coordinates(self, index, patch_shape, patch_stride, image_shape):
        # Unpack shapes
        patch_depth, patch_height, patch_width = patch_shape
        stride_depth, stride_height, stride_width = patch_stride
        image_depth, image_height, image_width = image_shape

        patches_per_row = (image_width - patch_width) // stride_width + 1
        patches_per_slice = patches_per_row * \
            ((image_height - patch_height) // stride_height + 1)

        z = index // patches_per_slice
        remainder = index % patches_per_slice
        y = (remainder // patches_per_row) * stride_height
        x = (remainder % patches_per_row) * stride_width

        start_z = min(z * stride_depth, image_depth - patch_depth)
        start_y = min(y, image_height - patch_height)
        start_x = min(x, image_width - patch_width)

        end_z = start_z + patch_depth
        end_y = start_y + patch_height
        end_x = start_x + patch_width

        return (start_z, start_y, start_x, end_z, end_y, end_x)

    def _infere(self, image: torch.Tensor, segmentation: torch.Tensor, is_train: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        prefix = "train" if is_train else "val"
        segmentation_hat = self.model(image)

        loss = self.loss(segmentation_hat,
                         segmentation, is_train=is_train, log_fn=self.log)
        loss += self.config.loss_config.xai_loss_weight * \
            self._xai_loss(segmentation_hat, segmentation, image, is_train)

        self.log(f"{prefix}/loss", loss.detach().cpu().item())
        return loss, segmentation_hat

    def _xai_loss(self, y_hat: torch.Tensor, y: torch.Tensor, input: torch.Tensor, is_train: bool = True) -> torch.Tensor:
        prefix = "train" if is_train else "val"
        cloned_hat = y_hat.clone()
        gradient = grad(cloned_hat, input, retain_graph=True)

        loss = self.overlap_loss(gradient, y)
        self.log(f"{prefix}/xai_loss", loss.detach().cpu().item())
        return loss

    def _get_from_batch(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        images = torch.concat(
            [i['data'] for k, i in batch.items() if k != 'label'], dim=1)
        segmentation = batch['label']['data'].squeeze()

        return images, segmentation

    def _get_patches(self, batch: torch.Tensor) -> Any:
        shape = batch.shape
        b_patch, c_patch, d_patch, h_patch, w_patch = patch_size = (
            shape[0], *self.config.data_config.patch_size
        )
        b_stride, c_stride, d_stride, h_stride, w_stride = (
            shape[0], *self.config.data_config.patch_strides
        )

        unfolded = batch.unfold(2, d_patch.item(), d_stride.item()).unfold(3, h_patch.item(), h_stride.item()).unfold(
            4, w_patch.item(), w_stride.item()).unfold(1, c_patch.item(), c_stride.item()).unfold(0, b_patch, b_stride)

        unfolded = unfolded.permute(
            0, 1, 9, 2, 3, 4, 5, 6, 7, 8).contiguous().view(-1, *patch_size)

        return unfolded

    def compute_overlapping_loss(self, y_hat: torch.Tensor, previous_y_hat: torch.Tensor, is_train: bool = True) -> torch.Tensor:
        mask = torch.zeros_like(y_hat, dtype=torch.bool)
        mask2 = torch.zeros_like(y_hat, dtype=torch.bool)
        strides = self.config.data_config.patch_strides
        overlap = self.config.data_config.image_size - \
            self.config.data_config.patch_strides
        mask[:, :, overlap[0]:, overlap[1]:, overlap[2]:] = True
        mask2[:, :, :strides[0], :strides[1], :strides[2]] = True

        overlap_loss = self.overlap_loss(y_hat[mask], previous_y_hat[mask])
        prefix = "train" if is_train else "val"
        self.log(f"{prefix}_overlap_loss", overlap_loss.detach().cpu().item())

        return overlap_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            self.config.loss_config.cosine_period,
            eta_min=self.config.loss_config.min_lr,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
        }
