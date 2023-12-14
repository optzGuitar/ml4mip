from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union
from diffusers.models.modeling_flax_utils import FlaxModelMixin
import flax.linen as nn
import jax.random as rnd
from torch.utils.data import Dataset
from diffusers.schedulers import FlaxSchedulerMixin
from optax import GradientTransformation
import jax


@dataclass
class DiffusionConfig:
    model: Union[nn.Module, FlaxModelMixin]
    scheduler: FlaxSchedulerMixin
    optimizer: GradientTransformation
    noise_type: Callable[[rnd.KeyArray, Sequence[int]], jax.Array]


@dataclass
class DataConfig:
    train_dataset: Dataset
    val_dataset: Optional[Dataset]
    test_dataset: Optional[Dataset]
    batch_size: int = 4
    epochs: int = 1

    num_workers: int = 4


@dataclass
class Config:
    diffusion: DiffusionConfig
    data: DataConfig
