from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
from segmentation.config import SegmentationConfig
from segmentation.segmentation import SegmentationModule
from data.segmentation import SegmentationDataset
import torch
from pytorch_lightning.loggers import WandbLogger


def train(config: SegmentationConfig):
    torch.manual_seed(config.seed)
    checkpoint_callback = ModelCheckpoint(
        dirpath="segmentation_checkpoints/", save_top_k=10, monitor="overall_mean", mode='max')
    trainer = Trainer(
        max_epochs=config.train_config.epochs,
        enable_checkpointing=True,
        logger=WandbLogger(project="ml4mip"),
        log_every_n_steps=1,
        accumulate_grad_batches=config.train_config.gradient_accumulation_steps,
        gradient_clip_val=config.loss_config.gradient_clip,
        callbacks=[checkpoint_callback]
    )
    model = SegmentationModule(config)
    dataset = SegmentationDataset(
        config, True)
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
