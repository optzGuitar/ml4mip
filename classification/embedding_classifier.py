from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch
import torch.nn as nn
import torchmetrics as tm
import torchvision.models as models


class MGMTClassifier(nn.Module):
    def __init__(self, num_cls=19, channels=1):
        super().__init__()
        self.resnet = models.resnet18()

        self.conv1 = nn.Conv2d(4, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.conv1 = self.conv1
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_cls)

    def forward(self, x):
        x = self.resnet(x)
        return x


class EmbeddingClassifier(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self._model = MGMTClassifier()

        self._loss = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self._model(x)
        loss = self._loss(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def on_train_epoch_end(self) -> None:
        torch.save(self._model.state_dict(), 'embedding_classifier.pkl')

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self._model(x)

        loss = self._loss(y_hat, y)
        acc = tm.functional.accuracy(y_hat, y, task="binary")
        prec = tm.functional.precision(y_hat, y, task="binary")
        rec = tm.functional.recall(y_hat, y, task="binary")
        f1 = tm.functional.f1_score(y_hat, y, task="binary")
        auc = tm.functional.auroc(y_hat, y, task="binary")

        self.log("val/loss", loss)
        self.log("val/acc", acc)
        self.log("val/prec", prec)
        self.log("val/rec", rec)
        self.log("val/f1", f1)
        self.log("val/auc", auc)

        return {"loss": loss, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, verbose=True)
        self._model = torch.jit.script(self._model)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}
