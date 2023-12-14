from diffusers import FlaxDDIMScheduler
from data.segmentation import SegmentationDataset
from diffusion_jax.config import Config, DiffusionConfig, DataConfig
from torch.utils.data import random_split
import torch
import jax.random as ra
import jax.numpy as jnp
import optax

from diffusion_jax.lr_scheduler import create_cosine
from diffusion_jax.model import FlaxUNet2DModel
from diffusion_jax.trainer import inference, train_model

key = ra.PRNGKey(42)
key, subkey = ra.split(key, 2)


# train_dataset, val_dataset, test_dataset = random_split(
#    SegmentationDataset(True, 5), (0.7, 0.1, 0.2))

class RandomDataset:
    def __init__(self, shape):
        self.shape = shape

    def __getitem__(self, index):
        return torch.ones(self.shape), torch.ones(self.shape)

    def __len__(self):
        return 100


train_dataset, val_dataset, test_dataset = random_split(
    RandomDataset((5, 224, 224)), (0.7, 0.1, 0.2))

data_config = DataConfig(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    batch_size=4,
    epochs=1,
    num_workers=0,
)

num_batches = len(train_dataset) // data_config.batch_size + \
    bool(len(train_dataset) % data_config.batch_size)

diffusion_config = DiffusionConfig(
    model=FlaxUNet2DModel(sample_size=224, in_channels=10,
                          out_channels=5),
    scheduler=FlaxDDIMScheduler(),
    optimizer=optax.adam(create_cosine(
        1e-3, 0, data_config.epochs, num_batches)),
    noise_type=ra.normal,
)

config = Config(
    diffusion=diffusion_config,
    data=data_config,
)

params = train_model(config)
sample = inference(model=config.diffusion.model, parameters=params, scheduler=config.diffusion.scheduler, image=jnp.asarray(
    test_dataset[0][0]), shape=jnp.asarray(test_dataset[0][1].shape), noise_generator=config.diffusion.noise_type, rng=key, debug=True)
