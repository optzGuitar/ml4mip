import os
import pickle

import torch
from data.classification import EmbeddedDataset  # noqa
import sys  # noqa
sys.path.insert(0, "/home/tu-leopinetzki/classif")  # noqa

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from classification.embedding_classifier import EmbeddingClassifier

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    model = EmbeddingClassifier()
    trainer = pl.Trainer(
        logger=WandbLogger(project="ml4mip"),
        callbacks=[ModelCheckpoint(monitor="f1")],
        log_every_n_steps=1,
        max_epochs=6,
    )

    dataset = EmbeddedDataset()
    train, val = random_split(dataset, [0.9, 0.1])
    dl = DataLoader(train, batch_size=64, shuffle=True, num_workers=4)
    dl_val = DataLoader(val, batch_size=256, shuffle=False, num_workers=4)

    trainer.fit(model, train_dataloaders=dl, val_dataloaders=dl_val)
