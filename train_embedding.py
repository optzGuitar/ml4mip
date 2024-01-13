import os
from data.classification import EmbeddingDataset  # noqa
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
    )

    dataset = EmbeddingDataset(full_augment=False, load_pickled=True)
    dl = DataLoader(dataset, batch_size=32)

    trainer.fit(model, train_dataloaders=dl)
