import pytorch_lightning as pl
from monai.networks.nets import resnet
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW


class ResNet50(pl.LightningModule):
    def __init__(
            self,
            spatial_dims=3,
            n_input_channels=4,
            num_classes=2,
            loss_fn=CrossEntropyLoss,
            learning_rate=0.01,
            weight_decay=0.1,
            max_epochs=50
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
        input_image, labels = batch      # !!!
        output = self.forward(input_image)
        loss = self.loss_fn(output, labels)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_image, labels = batch      # !!!
        output = self.forward(input_image)
        loss = self.loss_fn(output, labels)
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.wd
        )
        return optimizer