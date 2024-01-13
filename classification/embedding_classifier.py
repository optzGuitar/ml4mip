from typing import Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as tm


class MGMTClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._model = nn.Sequential(
            nn.Conv2d(4, 32, 5, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )
        self._head = nn.Sequential(nn.Linear(128, 2), nn.Softmax(dim=1))

    def forward(self, x):
        x = self._model(x)
        x = x.view(x.size(0), -1)
        x = self._head(x)
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
        acc = tm.functional.accuracy(y_hat, y)
        prec = tm.functional.precision(y_hat, y)
        rec = tm.functional.recall(y_hat, y)
        f1 = tm.functional.f1_score(y_hat, y)
        auc = tm.functional.auroc(y_hat, y)

        self.log("val/loss", loss)
        self.log("val/acc", acc)
        self.log("val/prec", prec)
        self.log("val/rec", rec)
        self.log("val/f1", f1)
        self.log("val/auc", auc)

        return {"loss": loss, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}
