import pytorch_lightning as pl
from torchvision import models
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch.nn as nn
from lightly.loss import BarlowTwinsLoss, NTXentLoss
import torchio as tio
import torch
from itertools import combinations
import pickle

from enums.contrast import ClassificationContrasts


class ResNet18(nn.Module):
    def __init__(self, num_cls=19, channels=1, pretrained=True):
        super().__init__()
        self.resnet = models.resnet18(pretrained=pretrained)

        self.conv1 = nn.Conv2d(channels, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.conv1 = self.conv1
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_cls)

    def forward(self, x):
        x = self.resnet(x)

        return x


class EmbeddingModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self._model = ResNet18(512, channels=1)

        self.loss = BarlowTwinsLoss()
        self.nxent_loss = NTXentLoss(memory_bank_size=1024)

    def training_step(self, batch, batch_idx):
        if batch_idx % 100 == 0:
            with open('embedding.pkl', 'wb') as f:
                pickle.dump(self._model.state_dict(), f)

        z0 = self._model(batch[:, 0:1])
        z1 = self._model(batch[:, 1:2])
        z2 = self._model(batch[:, 2:3])
        z3 = self._model(batch[:, 3:4])

        i = 1
        loss = torch.zeros(1, device=batch.device)
        for comb in combinations([z0, z1, z2, z3], 2):
            loss += self.loss(comb[0], comb[1])
            loss += self.nxent_loss(comb[0], comb[1])
            i += 1

        loss /= i

        self.log("train/loss", loss)
        return loss

    def on_train_epoch_end(self) -> None:
        with open('embedding_end.pkl', 'wb') as f:
            pickle.dump(self._model.state_dict(), f)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.model = torch.jit.script(self._model)
        return optimizer
