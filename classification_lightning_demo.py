from pytorch_lightning import Trainer
from torch.utils.data import random_split
from classification.resnet import ResNet50
from data.classification import ClassificationDataset
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import CSVLogger
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

batch_size = 1

n = len(os.listdir("/data/classification/train/"))
train_size = int(0.9*n)
valid_size = n - train_size 

train_ds = ClassificationDataset(full_augment=False, load_pickled=False)
train_ds, valid_ds = random_split(train_ds, [train_size, valid_size])

train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=7)
valid_dataloader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=7)

model = ResNet50(spatial_dims=3, num_classes=2, max_epochs=1)

logger = CSVLogger(save_dir="logs/", name="resnet")

trainer = Trainer(
    accelerator="gpu",
    devices=[1],
    max_epochs=model.max_epochs,
    logger=logger
)

if __name__ == '__main__':
    trainer.fit(model, train_dataloader, valid_dataloader)