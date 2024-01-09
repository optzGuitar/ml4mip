import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from convolution import convNd, BatchNorm4d


class Branch(nn.Module):
    def __init__(self, input_channels, filters, kernel_size, pad=(0,0,0,0)):
        super().__init__()
        self.input_channels = input_channels
        self.middle_channels = input_channels*filters
        self.output_channels = self.middle_channels*filters
        
        self.conv1 = convNd(
            in_channels=self.input_channels,
            out_channels=self.middle_channels,
            num_dims=4,
            kernel_size=kernel_size,
            stride=1,
            padding=pad
            )
        self.conv2 = convNd(
            in_channels=self.middle_channels,
            out_channels=self.output_channels,
            num_dims=4,
            kernel_size=kernel_size,
            stride=1,
            padding=pad
            )
        self.batchNorm = BatchNorm4d(
            self.middle_channels,
            eps=0.001,
            momentum=0.99
            )
        self.flatten = nn.Flatten()
        self.activation_function = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.activation_function(out)
        
        out = self.batchNorm(out)
        
        out = self.conv2(out)
        out = self.activation_function(out)
        
        out = self.flatten(out)
        return out


class OutputBranch(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.activation_function = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.linear1 = nn.Linear(in_features, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, num_classes)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.activation_function(out)
        out = self.dropout1(out)
        
        out = self.linear2(out)
        out = self.activation_function(out)
        out = self.dropout2(out)
        
        out = self.linear3(out)
        out = self.softmax(out)
        return out


class TriCNN(nn.Module):
    def __init__(self, input_img_size, input_channels, filters, num_classes):   # Change input_channels depending on PCA or not
        super().__init__()
        h, w, d, fourth_dim = input_img_size
        in_features = filters*filters*fourth_dim*(h*w*d + 2*(h-4)*(w-4)*(d-4))
        
        self.spatialBranch = Branch(input_channels, filters, (3,3,3,1))
        self.spectralBranch = Branch(input_channels, filters, (1,1,1,3), pad=(0,0,0,1))
        self.spatialSpectralBranch = Branch(input_channels, filters, (3,3,3,3), pad=(0,0,0,1))
        self.outputBranch = OutputBranch(in_features, num_classes)
    
    def forward(self, x):
        y1 = self.spatialBranch(x)
        y2 = self.spectralBranch(x)
        y3 = self.spatialSpectralBranch(x)
        
        out = torch.cat((y1, y2, y3), dim=1)
        
        out = self.outputBranch(out)
        return out


class TriCNN_Lightning(pl.LightningModule):
    def __init__(
            self,
            input_image_size,
            input_channels=1,
            filters=64,
            num_classes=2,
            loss_fn=CrossEntropyLoss(),
            learning_rate=0.001,
            weight_decay=0.1,
            max_epochs=50
        ):
        super().__init__()
        self.model = TriCNN(input_image_size, input_channels, filters, num_classes)
        self.loss_fn = loss_fn
        self.lr = learning_rate
        self.wd = weight_decay
        self.max_epochs = max_epochs
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        input_image, labels = batch
        output = self.forward(input_image)
        loss = self.loss_fn(output, labels)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_image, labels = batch
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