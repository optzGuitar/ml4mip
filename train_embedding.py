from data.classification import ClassificationDataset  # noqa
import sys  # noqa
sys.path.insert(0, "/home/tu-leopinetzki/classif")  # noqa

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from classification.embedding import EmbeddingModule

if __name__ == '__main__':
    model = EmbeddingModule()
    trainer = pl.Trainer()

    dataset = ClassificationDataset(full_augment=False, load_pickled=True)
    dl = DataLoader(dataset, batch_size=4)

    trainer.fit(model, train_dataloaders=dl)
