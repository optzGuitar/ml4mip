import os
import sys
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint

from data.classification import ClassificationDataset  # noqa
sys.path.insert(0, "/home/tu-leopinetzki/ml4mip")  # noqa

from classification.resnet import ResNet50

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    model = ResNet50(max_epochs=20)

    train_ds = ClassificationDataset(full_augment=False)
    train_ds, valid_ds = random_split(train_ds, [0.9, 0.1])

    BATCH_SIZE = 16

    train_dataloader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(
        valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    checkpoint_callback = ModelCheckpoint(
        dirpath="classification_checkpoints/", save_top_k=10, monitor="val/f1", mode='max')

    trainer = Trainer(
        max_epochs=model.max_epochs,
        logger=WandbLogger(project="ml4mip"),
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=4,
    )
    trainer.fit(model, train_dataloader, valid_dataloader)
