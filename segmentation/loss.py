import torch.nn as nn
import torch.nn.functional as F
import torch
import segmentation_models_pytorch as smp
from pytorch_lightning.loggers import WandbLogger

from segmentation.config import SegmentationConfig


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1e-6
        self.axes = (2, 3, 4)

    def forward(self, y_true, y_pred):
        # Prepare inputs
        y_pred = F.softmax(y_pred, dim=1)

        num = torch.sum(torch.square(y_true - y_pred), dim=self.axes)
        den = torch.sum(torch.square(y_true), dim=self.axes) + torch.sum(torch.square(y_pred),
                                                                         dim=self.axes) + self.smooth

        loss = torch.mean(num / den, axis=1)
        loss = torch.mean(loss)
        return loss


class DiceCELoss(nn.Module):
    def __init__(self):
        super(DiceCELoss, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, y_true, y_pred):
        # Dice loss
        loss_dice = self.dice_loss(y_true, y_pred)

        # Cross entropy loss
        loss_ce = self.cross_entropy(y_pred, y_true)

        return loss_ce + loss_dice


class GenSurfLoss(nn.Module):
    def __init__(self, class_weights=None):
        super(GenSurfLoss, self).__init__()
        self.region_loss = DiceCELoss()

        # Define class weight scheme
        # Move weights to cuda if already given by user
        if not (class_weights is None):
            self.class_weights = torch.Tensor(class_weights).to("cuda")
        else:
            self.class_weights = None

        self.smooth = 1e-6
        self.axes = (2, 3, 4)

    def forward(self, y_true, y_pred, dtm, alpha):
        # Compute region based loss
        region_loss = self.region_loss(y_true, y_pred)

        y_pred = F.softmax(y_pred, dim=1)

        if self.class_weights is None:
            class_weights = torch.sum(y_true, dim=self.axes)
            class_weights = 1. / (torch.square(class_weights) + 1.)
        else:
            class_weights = self.class_weights

        # Compute boundary loss
        # Flip each one-hot encoded class
        y_worst = torch.square(1.0 - y_true)

        num = torch.sum(torch.square(dtm * (y_worst - y_pred)), axis=self.axes)
        num *= class_weights

        den = torch.sum(torch.square(dtm * (y_worst - y_true)), axis=self.axes)
        den *= class_weights
        den += self.smooth

        boundary_loss = torch.sum(num, axis=1) / torch.sum(den, axis=1)
        boundary_loss = torch.mean(boundary_loss)
        boundary_loss = 1. - boundary_loss

        return alpha * region_loss + (1. - alpha) * boundary_loss


class CustomLoss(nn.Module):
    def __init__(self, config: SegmentationConfig, ce_weight: float = 1, tversky_weight: float = 1, gsl_weight: float = 1) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.tversky_weight = tversky_weight
        self.gsl_weight = gsl_weight
        self.ce = nn.CrossEntropyLoss()
        self.tversky_loss = smp.losses.TverskyLoss(
            mode="multiclass",
            alpha=config.loss_config.tversky_alpha, beta=config.loss_config.tversky_beta)
        self.gsl_loss = GenSurfLoss()
        self.config = config

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, is_train: bool = True, log_fn=None) -> torch.Tensor:
        y_index = torch.argmax(y, dim=1)
        ce_loss = self.ce(y_hat.view(
            y_hat.shape[0], -1, y_hat.shape[1]), y_index.view(y.shape[0], -1))
        tversky_loss = self.tversky_loss(
            y_hat, y_index)
        gsl_loss = self.gsl_loss(
            y_hat, y, dtm=self.config.loss_config.dtm, alpha=self.config.loss_config.alpha)
        combined = (
            self.ce_weight * ce_loss +
            self.tversky_weight * tversky_loss +
            self.gsl_weight * gsl_loss
        )

        if log_fn is not None:
            prefix = "train" if is_train else "val"
            log_fn(
                f"{prefix}/ce_loss", ce_loss.detach().cpu().mean().item())
            log_fn(
                f"{prefix}/tversky_loss", tversky_loss.detach().cpu().mean().item())
            log_fn(
                f"{prefix}/gsl_loss", gsl_loss.detach().cpu().mean().item())
            log_fn(
                f"{prefix}/custom_loss", combined.detach().cpu().mean().item())

        return combined.mean()
