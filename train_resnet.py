import os
import sys
# sys.path.insert(0, "/home/tu-leopinetzki/classif")  # noqa
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from data.classification import ClassificationDataset
from classification.resnet import ResNet50

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    model = ResNet50(max_epochs=20)

    train_ds = ClassificationDataset(full_augment=False, load_pickled=True)
    train_ds, valid_ds = random_split(train_ds, [0.9, 0.1])

    BATCH_SIZE = 4

    train_dataloader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(
        valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    trainer = Trainer(
        max_epochs=model.max_epochs,
        logger=WandbLogger(project="ml4mip"),
        accumulate_grad_batches=8,
        log_every_n_steps=1,
    )
    trainer.fit(model, train_dataloader, valid_dataloader)
