import sys  # noqa
sys.path.insert(0, "/home/tu-leopinetzki/ml4mip")  # noqa

from pytorch_lightning.loggers import WandbLogger
import torch
from data.segmentation import SegmentationDataset
from segmentation.segmentation import SegmentationModule
from segmentation.config import SegmentationConfig, TrainConfig, LossConfig, DataConfig
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer


def train():
    config = SegmentationConfig(
        run_name="SegResNetAccCEXAI",
        train_config=TrainConfig(
            epochs=50,
            gradient_accumulation_steps=32,
            batch_size=2,
        ),
        loss_config=LossConfig(
            cosine_period=200
        ),
        data_config=DataConfig(load_pickle=False)
    )
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
