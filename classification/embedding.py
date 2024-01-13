import pytorch_lightning as pl
from torchvision import models
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch.nn as nn
from lightly.loss import BarlowTwinsLoss
import torchio as tio
import torch
from itertools import combinations
import pickle

from enums.contrast import ClassificationContrasts


class ResNet18(nn.Module):
    def __init__(self, num_cls=19, channels=1, pretrained=True):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)

        self.conv1 = nn.Conv2d(channels, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resnet.conv1 = self.conv1
        resnet.fc = nn.Linear(resnet.fc.in_features, num_cls)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        logits = self.FC(x)

        return logits


class EmbeddingModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self._model = ResNet18(128, channels=1)

        self.loss = BarlowTwinsLoss()

    def training_step(self, batch, batch_idx):
        if batch_idx % 100 == 0:
            with open('embedding.pkl', 'wb') as f:
                pickle.dump(self._model.state_dict(), f)

        loss = torch.zeros(1, device=batch.device)
        for comb in combinations([0, 1, 2, 3], 2):
            z0 = self._model(batch[:, comb[0]:comb[0]+1])
            z1 = self._model(batch[:, comb[1]:comb[1]+1])
            loss += self.loss(z0, z1)
            torch.cuda.empty_cache()

        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.model = torch.jit.script(self._model)
        return optimizer
