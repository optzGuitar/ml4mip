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
            spatial_dims=2,
            n_input_channels=1
        )
        self._model.fc = nn.Linear(self._model.fc.in_features, 128)

        self.loss = BarlowTwinsLoss()

    def training_step(self, batch, batch_idx):
        if batch_idx % 100 == 0:
            with open('embedding.pkl', 'wb') as f:
                pickle.dump(self._model.state_dict(), f)

        loss = torch.zeros(1, device=batch.device)
        for comb in combinations([0, 1, 2, 3], 2):
            z0 = self._model(batch[:, comb[0]:comb[0]+1])
            z1 = self._model(batch[:, comb[0]:comb[0]+1])
            loss += self.loss(z0, z1)
            torch.cuda.empty_cache()

        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.model = torch.jit.script(self._model)
        return optimizer
