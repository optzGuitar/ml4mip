import pytorch_lightning as pl
from monai.networks.nets import resnet
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch.nn as nn
from lightly.loss import BarlowTwinsLoss
import torchio as tio
import torch
from itertools import combinations
import pickle

from enums.contrast import ClassificationContrasts


class EmbeddingModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self._model = resnet.resnet18(
            num_classes=1,
            spatial_dims=3,
            n_input_channels=1
        )
        self._model.fc = nn.Identity()

        self.loss = BarlowTwinsLoss()

    def training_step(self, batch, batch_idx):
        if batch_idx % 100 == 0:
            with open('embedding.pkl', 'wb') as f:
                pickle.dump(self._model.state_dict(), f)

        z0 = self._model(batch[:, 0:1])
        z1 = self._model(batch[:, 1:2])
        z2 = self._model(batch[:, 3:4])
        z3 = self._model(batch[:, 4:5])

        loss = torch.zeros(1, device=z0.device)
        for comb in combinations([z0, z1, z2, z3], 2):
            loss += self.loss(comb[0], comb[1])

        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.model = torch.jit.script(self._model)
        return optimizer
