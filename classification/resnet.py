import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import torch
import torchio as tio
import torchmetrics as tm
import torch.nn as nn
import torchvision.models as models


class Block(nn.Module):
    def __init__(self, input_channels, filters, kernel_size, flatten=True):
        super().__init__()
        stride = 2
        padding = ((kernel_size - 1) * (stride - 1)) // 2
        self.should_flatten = flatten

        self.channels = input_channels

        self.conv1 = nn.Conv3d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.conv2 = nn.Conv3d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.batchNorm = nn.BatchNorm3d(
            self.channels,
            eps=0.001,
            momentum=0.99
        )
        self.flatten = nn.Flatten()
        self.activation_function = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.activation_function(out)

        out = self.batchNorm(out)

        out = self.conv2(out)
        out = self.activation_function(out)

        if self.should_flatten:
            out = out + x
            out = out.reshape(out.shape[0], out.shape[1], out.shape[2], -1)
        return out


class ResNet(nn.Module):
    def __init__(self, num_cls=19):
        super().__init__()
        self.resnet = models.resnet34()

        self.resnet.conv1 = Block(4, 32, 7)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_cls)

    def forward(self, x):
        x = self.resnet(x)

        return x


class ResNet50(pl.LightningModule):
    def __init__(
        self,
        learning_rate=0.001,
        weight_decay=0.1,
        max_epochs=1
    ):
        super().__init__()
        self.model = ResNet(2)
        self.loss_fn = CrossEntropyLoss()
        self.lr = learning_rate
        self.wd = weight_decay
        self.max_epochs = max_epochs

    def forward(self, x):
        return self.model(x)

    def _split_batch(self, batch):
        input_images = torch.cat(
            [i[tio.DATA] for k, i in batch.items() if k != tio.LABEL], dim=1)
        label = batch[tio.LABEL]
        mask = input_images.isnan()
        if mask.any():
            input_images[mask] = 0
        return input_images, label

    def training_step(self, batch, batch_idx):
        input_images, label = self._split_batch(batch)
        output = self.forward(input_images)
        loss = self.loss_fn(output, label)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_images, label = self._split_batch(batch)
        output = self.forward(input_images)
        loss = self.loss_fn(output, label)

        acc = tm.functional.accuracy(output, label)
        prec = tm.functional.precision(output, label)
        rec = tm.functional.recall(output, label)
        f1 = tm.functional.f1(output, label)
        auc = tm.functional.auroc(output, label)

        self.log("val/loss", loss)
        self.log("val/acc", acc)
        self.log("val/prec", prec)
        self.log("val/rec", rec)
        self.log("val/f1", f1)
        self.log("val/auc", auc)

        return {"loss": loss, "acc": acc, "prec": prec, "rec": rec, "val/f1": f1, "auc": auc}

    def on_train_epoch_end(self) -> None:
        torch.save(self.model.state_dict(), 'resnet50_end.pkl')

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.wd
        )
        self.model = torch.jit.script(self.model)
        return optimizer
