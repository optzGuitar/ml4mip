import pytorch_lightning as pl
from torchvision.models import resnet18
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

        self._model = resnet18(
            num_classes=1,
            n_input_channels=1
        )
        self._model.fc = nn.Identity()

        self.loss = BarlowTwinsLoss()

    def training_step(self, batch, batch_idx):
        if batch_idx % 100 == 0:
            with open('embedding.pkl', 'wb') as f:
                pickle.dump(self._model.state_dict(), f)

        input, _ = batch
        z0 = self._model(input[:, 0:1])
        z1 = self._model(input[:, 1:2])
        z2 = self._model(input[:, 3:4])
        z3 = self._model(input[:, 4:5])

        loss = torch.zeros(1, device=z0.device)
        for comb in combinations([z0, z1, z2, z3], 2):
            loss += self.loss(comb[0], comb[1])

        self.log("train/loss", loss)
        return loss

    def _split_batch(self, batch):
        input_images = torch.cat(
            [batch[contrast][tio.DATA] for contrast in ClassificationContrasts.values()], dim=1)
        label = batch[tio.LABEL][tio.DATA]

        return input_images, label
