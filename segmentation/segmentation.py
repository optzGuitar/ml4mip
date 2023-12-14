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


class SegmentationModule(pl.LightningModule):
    def __init__(self, segmentation_config: SegmentationConfig):
        super().__init__()
        self.model = nets.attentionunet.AttentionUnet(
            spatial_dims=segmentation_config.data_config.n_dims,
            in_channels=segmentation_config.data_config.n_channels,
            out_channels=segmentation_config.data_config.n_channels,
            channels=segmentation_config.model_config.channels,
            strides=segmentation_config.model_config.strides,
            kernel_size=segmentation_config.model_config.kernels,
            up_kernel_size=list(
                reversed(segmentation_config.model_config.kernels)),
            dropout=segmentation_config.model_config.dropout,
        )

        self.loss = CustomLoss(
            segmentation_config,
            ce_weight=segmentation_config.loss_config.ce_weight,
            tversky_weight=segmentation_config.loss_config.tversky_loss,
            gsl_weight=segmentation_config.loss_config.gen_surf_weight,
            logger=self.logger,
        )
        self.overlap_loss = nn.MSELoss()

        self.config = segmentation_config

        self.save_hyperparameters('segmentation_config')
        self.automatic_optimization = False

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        optimizer = self.optimizers()
        lr_scheduler = self.lr_schedulers()

        images, segmentation = self._get_from_batch(batch)

        image_patches = self._get_patches(images)
        segmentation_patches = self._get_patches(segmentation)

        previous_segmentation_hat_p = [None]
        for loss, _ in self._handle_patch_batch(image_patches, segmentation_patches, previous_segmentation_hat_p):
            self.manual_backward(loss)

        self.clip_gradients(
            optimizer,
            gradient_clip_val=self.config.loss_config.gradient_clip,
            gradient_clip_algorithm="norm"
        )
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        images, segmentation = self._get_from_batch(batch)
        image_patches = self._get_patches(images)
        segmentation_patches = self._get_patches(segmentation)

        previous_seg_hat_p = [None]
        segmentations_hat = torch.zeros_like(segmentation)
        for i, (loss, segmentaiton_hat) in enumerate(self._handle_patch_batch(image_patches, segmentation_patches, previous_seg_hat_p, is_train=False)):
            # TODO: add image logging!
            start_z, start_y, start_x, end_z, end_y, end_x = self.flatten_index_to_coordinates(
                i, self.config.data_config.patch_size, self.config.data_config.strides, self.config.data_config.image_size
            )
            segmentations_hat[:, :, start_z:end_z,
                              start_y:end_y, start_x:end_x
                              ] = segmentaiton_hat

        dice = MulticlassF1Score(
            average=None, num_classes=self.config.data_config.n_classes - 1).to(self.device)
        dice_score = dice(
            segmentations_hat[:, 1:], segmentation[:, 1:])
        tumor_score = compute_hausdorff_distance(
            segmentations_hat[:, [1, 4]], segmentation[:,
                                                       :, [1, 4], :, :], include_background=True
        )
        whole_tumor = compute_hausdorff_distance(
            segmentations_hat[:, [1, 2, 4]], segmentation[:,
                                                          :, [1, 2, 4]], include_background=True
        )

        for i in ['necrotic', 'edematous', 'enahncing']:
            self.log(f"val/dice_{i}", dice_score[i].mean().item())

        self.log("val/tumor_score", tumor_score.mean().item())
        self.log("val/whole_tumor", whole_tumor.mean().item())

        return (dice_score.mean() + tumor_score.mean() + whole_tumor.mean()).mean()

    def flatten_index_to_coordinates(index, patch_shape, patch_stride, image_shape):
        # Unpack shapes
        patch_depth, patch_height, patch_width, _ = patch_shape
        stride_depth, stride_height, stride_width = patch_stride
        image_depth, image_height, image_width, _ = image_shape

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

    def _handle_patch_batch(self, images: torch.Tensor, segmentation: torch.Tensor, previous_seg_hat_p: list[Optional[torch.Tensor]], is_train: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        prefix = "train" if is_train else "val"
        for image_patch, segmentation_patch in zip(images, segmentation):
            segmentation_hat = self.model(image_patch)
            loss = self.loss(segmentation_hat,
                             segmentation_patch, is_train=is_train)

            if previous_seg_hat_p[0] is not None:
                loss += (self.config.loss_config.patch_loss_weight *
                         self.overlap_loss(
                             segmentation_hat,
                             previous_seg_hat_p[0],
                             is_train=is_train,
                         ))

            previous_seg_hat_p[0] = segmentation_hat

            self.log(f"{prefix}/loss", loss.detach().cpu().item())
            yield loss, segmentation_hat

    def _get_from_batch(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        images = torch.concat(
            [i['data'] for k, i in batch.items() if k != 'label'], dim=1)
        segmentation = batch['label']['data'].squeeze()

        return images, segmentation

    def _get_patches(self, batch: torch.Tensor) -> Any:
        shape = batch.shape
        b_patch, c_patch, d_patch, h_patch, w_patch = patch_size = (
            shape[0], shape[1], *self.config.data_config.patch_size
        )
        b_stride, c_stride, d_stride, h_stride, w_stride = (
            shape[0], shape[1], *self.config.data_config.strides
        )

        unfolded = batch.unfold(2, d_patch, d_stride).unfold(3, h_patch, h_stride).unfold(
            4, w_patch, w_stride).unfold(1, c_patch, c_stride).unfold(0, b_patch, b_stride)

        unfolded = unfolded.permute(
            0, 1, 9, 2, 3, 4, 5, 6, 7, 8).contiguous().view(-1, *patch_size)

        return unfolded

    def compute_overlapping_loss(self, y_hat: torch.Tensor, previous_y_hat: torch.Tensor, is_train: bool = True) -> torch.Tensor:
        mask = torch.zeros_like(y_hat, dtype=torch.bool)
        mask2 = torch.zeros_like(y_hat, dtype=torch.bool)
        strides = self.config.data_config.strides
        overlap = self.config.data_config.image_size - self.config.data_config.strides
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
