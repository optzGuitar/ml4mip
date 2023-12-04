from pytorch_lightning import Trainer
from torch.utils.data import random_split
from classification.model import ResNet50
from data.classification import ClassificationDataset
from torch.utils.data import DataLoader
import os

batch_size = 4

n = len(os.listdir("data/classification/train/"))
train_size = int(0.9*n)
valid_size = n - train_size

train_ds = ClassificationDataset(full_augment=False)
train_ds, valid_ds = random_split(train_ds, [train_size, valid_size])

train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

model = ResNet50(spatial_dims=3, num_classes=2)

trainer = Trainer(
    accelerator="gpu",
    devices=[1],
    max_epochs=model.max_epochs
)
trainer.fit(model, train_dataloader, valid_dataloader)