from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, random_split
from segmentation.config import SegmentationConfig
from segmentation.segmentation import SegmentationModule
from data.segmentation import SegmentationDataset
import torch
from pytorch_lightning.loggers import WandbLogger


def train(config: SegmentationConfig):
    torch.manual_seed(config.seed)
    trainer = Trainer(
        default_root_dir=config.root_dir,
        max_epochs=config.train_config.epochs,
        enable_checkpointing=True,
        logger=WandbLogger(project="ml4mip")
    )
    model = SegmentationModule(config)
    dataset = SegmentationDataset(config, True, config.data_config.n_classes)
    train_dataset, val_dataset = random_split(dataset, (0.95, 0.05))

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.train_config.batch_size, num_workers=config.train_config.num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.train_config.batch_size, num_workers=config.train_config.num_workers
    )

    trainer.fit(model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
                )
