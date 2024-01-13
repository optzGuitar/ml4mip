import pytorch_lightning as pl
from monai.networks.nets import resnet
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from enums.contrast import ClassificationContrasts
import torch
import torchio as tio
import torchmetrics as tm
import torch.nn as nn
import torchvision.models as models


class Block(nn.Module):
    def __init__(self, input_channels, filters, kernel_size, flatten=False):
        super().__init__()
        stride = 2
        padding = ((kernel_size - 1) * (stride - 1)) // 2
        self.flatten = flatten

        self.input_channels = input_channels
        self.middle_channels = input_channels*filters
        self.output_channels = self.middle_channels*filters

        self.conv1 = nn.Conv3d(
            in_channels=self.input_channels,
            out_channels=self.middle_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.conv2 = nn.Conv3d(
            in_channels=self.middle_channels,
            out_channels=self.output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.batchNorm = nn.BatchNorm3d(
            self.middle_channels,
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

        if self.flatten:
            out = out.reshape(out.shape[0], out.shape[1], out.shape[2], -1)
        return out + x


class ResNet50(nn.Module):
    def __init__(self, num_cls=19, channels=1):
        super().__init__()
        self.resnet = models.resnet50()

        self.resnet.conv1 = nn.Sequential(
            Block(4, 32, 7, False), Block(32, 32, 5))
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_cls)

    def forward(self, x):
        x = self.resnet(x)

        return x


class ResNet50(pl.LightningModule):
    def __init__(
        self,
        spatial_dims=3,
        n_input_channels=4,
        num_classes=2,
        loss_fn=CrossEntropyLoss(),
        learning_rate=0.001,
        weight_decay=0.1,
        max_epochs=1
    ):
        super().__init__()
        self.model = resnet.resnet50(
            spatial_dims=spatial_dims,
            num_classes=num_classes,
            n_input_channels=n_input_channels
        )
        self.loss_fn = loss_fn
        self.lr = learning_rate
        self.wd = weight_decay
        self.max_epochs = max_epochs

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input_images = torch.cat(
            [i[tio.DATA] for k, i in batch.items() if k != tio.LABEL], dim=1)
        label = batch[tio.LABEL]
        output = self.forward(input_images)
        loss = self.loss_fn(output, label)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_images = torch.cat(
            [i[tio.DATA] for k, i in batch.items() if k != tio.LABEL], dim=1)
        label = batch[tio.LABEL]
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

        return {"loss": loss, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.wd
        )
        self.model = torch.jit.script(self.model)
        return optimizer
