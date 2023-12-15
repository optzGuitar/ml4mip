from dataclasses import dataclass, field
import torch


@dataclass
class DataConfig:
    n_dims: int = 3
    n_channels: int = 4
    n_classes: int = 4

    image_size: tuple = field(
        default_factory=lambda: torch.Tensor((4, 224, 224, 128)).to(torch.int))
    patch_size: tuple = field(
        default_factory=lambda: torch.Tensor((4, 224 // 4, 224 // 4, 128 // 4)).to(torch.int))
    patch_strides: tuple = field(default_factory=lambda: torch.Tensor(
        (4, 224 // 8, 224 // 8, 128 // 8)).to(torch.int)
    )


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 1
    num_workers: int = 4
    gradient_accumulation_steps: int = 1


@dataclass
class LossConfig:
    ce_weight: float = 1.0
    tversky_loss: float = 1.0
    gen_surf_weight: float = 1.0
    patch_loss_weight: float = 0.2

    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7

    cosine_period: int = 10000
    min_lr: float = 1e-6
    gradient_clip: float = 2.0


@dataclass
class ModelConfig:
    channels: list = field(default_factory=lambda: [32, 64, 128, 256])
    strides: list = field(default_factory=lambda: [3, 3, 2, 2])
    kernels: list = field(default_factory=lambda: [7, 5, 3, 3])
    dropout: float = 0.1

    number_patches: int = 4
    amount_overlap: float = 0.5


@dataclass
class SegmentationConfig:
    root_dir: str = "segmentation_logs"
    run_name: str = "segmentation"
    seed: int = 0
    data_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    loss_config: LossConfig = field(default_factory=LossConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
