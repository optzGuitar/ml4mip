from classification.tricnn_model import TriCNN_Lightning
from pytorch_lightning import Trainer
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset
from data.classification import ClassificationDataset
import torch
import os


os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

batch_size = 4

n = len(os.listdir("/data/classification/train/"))

train_ds = ClassificationDataset(full_augment=False)
train_ds, valid_ds = random_split(train_ds, [0.8, 0.2])

train_dataloader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
valid_dataloader = DataLoader(
    valid_ds, batch_size=batch_size, shuffle=False, num_workers=7)

model = TriCNN_Lightning(
    (15, 15, 15, 4),
    filters=4
)

trainer = Trainer(
    devices=[1],
    max_epochs=model.max_epochs
)

if __name__ == '__main__':
    trainer.fit(model, train_dataloader, valid_dataloader)
