import os
import sys  # noqa
sys.path.insert(0, "/home/tu-leopinetzki/ml4mip")  # noqa

from pytorch_lightning.loggers import WandbLogger
import torch
from data.segmentation_data import SegmentationDataset
from segmentation.seg_module import SegModule
from segmentation.config import SegmentationConfig, TrainConfig, LossConfig, DataConfig
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import torchio as tio


def train():
    config = SegmentationConfig(
        run_name="tio_pipeline_fr",
        train_config=TrainConfig(
            epochs=50,
            gradient_accumulation_steps=16,
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
        callbacks=[checkpoint_callback],
    )
    model = SegModule(config)
    if os.path.exists(f"segmentation_checkpoints/{config.run_name}_last.ckpt"):
        model = SegModule.load_from_checkpoint(
            f"segmentation_checkpoints/{config.run_name}_last.ckpt", segmentation_config=config)
    dataset = SegmentationDataset(
        config, True)
    train_dataset, val_dataset = random_split(dataset, (0.95, 0.05))

    # sampler = tio.data.LabelSampler(
    #     patch_size=config.data_config.patch_size, label_probabilities=[0.1, 0.3, 0.3, 0.3])
    # queue = tio.Queue(
    #     train_dataset,
    #     100,
    #     9,
    #     sampler,
    #     shuffle_subjects=True,
    #     num_workers=config.train_config.num_workers,
    # )

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.train_config.batch_size, num_workers=0
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, num_workers=config.train_config.num_workers
    )

    trainer.fit(model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
                )


if __name__ == "__main__":
    train()
