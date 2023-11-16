from typing import Any
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.losses as smp_losses
import torch.nn as nn
import torch
import wandb

from losses.loss_wrapper import LossWrapper


class CombinationLoss(LossWrapper):
    def __init__(self, log_individual_losses: bool) -> None:
        super().__init__(log_individual_losses=log_individual_losses)
        # for alpha, beta = 0.5 same as dice loss
        self._tversky_loss = smp_losses.TverskyLoss(
            mode=smp_losses.MULTILABEL_MODE, smooth=1e-6)
        self._lovasz_loss = smp_losses.LovaszLoss(
            mode=smp_losses.MULTILABEL_MODE)
        self._cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.05)

    def forward(self, X_hat: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        tversky_loss = self._tversky_loss(X_hat, X)
        lovasz_loss = self._lovasz_loss(X_hat, X)
        ce = self._cross_entropy(X_hat, X)

        if self._log_individual_losses:
            wandb.log(
                {
                    f"{self._prefix}tversky_loss": tversky_loss,
                    f"{self._prefix}lovasz_loss": lovasz_loss,
                    f"{self._prefix}cross_entropy": ce
                },
                step=self._step
            )

        return tversky_loss + lovasz_loss + ce
