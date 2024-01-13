import os
import pickle

import torch
from data.classification import ClassificationDataset, EmbeddingDataset  # noqa
import sys  # noqa
sys.path.insert(0, "/home/tu-leopinetzki/classif")  # noqa

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from classification.embedding import EmbeddingModule

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    model = EmbeddingModule()
    trainer = pl.Trainer(
        accumulate_grad_batches=4,
        logger=WandbLogger(project="ml4mip"),
        log_every_n_steps=1,
        max_epochs=7,
    )

    dataset = EmbeddingDataset(full_augment=False, load_pickled=True)
    dl = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

    trainer.fit(model, train_dataloaders=dl)

    resnet = model._model
    resnet.eval()

    with torch.no_grad():
        dataset = ClassificationDataset(full_augment=False, load_pickled=True)
        loader = DataLoader(dataset, batch_size=16,
                            shuffle=False, num_workers=4)
        for i, batch in enumerate(loader):
            data = batch.to(model.device)

            embedded = []
            slice = slice.permute(4, 0, 1, 2, 3)
            orig_shape = slice.shape
            slice = slice.view(
                orig_shape[0] * orig_shape[1], *orig_shape[2:])

            emb0 = resnet(slice[:, 0:1]).view(
                orig_shape[0], orig_shape[1], -1)
            emb1 = resnet(data[:, 1:2]).view(
                orig_shape[0], orig_shape[1], -1)
            emb2 = resnet(data[:, 2:3]).view(
                orig_shape[0], orig_shape[1], -1)
            emb3 = resnet(data[:, 3:4]).view(
                orig_shape[0], orig_shape[1], -1)

            emb0 = emb0.permute(1, 2, 0)
            emb1 = emb1.permute(1, 2, 0)
            emb2 = emb2.permute(1, 2, 0)
            emb3 = emb3.permute(1, 2, 0)

            for x, image in enumerate(torch.stack([emb0, emb1, emb2, emb3], dim=1)):
                embedded = image.cpu()
                with open(f"embeddings/embedding_{(i*16) + x}.pkl", "wb") as f:
                    pickle.dump(embedded, f)
