from typing import Any, Callable
from diffusion_jax.config import Config
from torch.utils.data import Dataset, DataLoader
import flax.linen as nn
import jax.random as rnd
import jax.numpy as jnp
from jax.random import KeyArray
import jax
import optax
from flax.training import train_state
import wandb
from diffusers.schedulers.scheduling_utils_flax import FlaxSchedulerMixin
from tqdm import tqdm


def train_model(config: Config) -> jax.Array:
    model = config.diffusion.model
    scheduler = config.diffusion.scheduler
    scheduler_state = scheduler.create_state()
    noise_generator = config.diffusion.noise_type
    optimizer = config.diffusion.optimizer

    num_timesteps = scheduler.config["num_train_timesteps"]

    # wandb.init(project='ml4mip')
    rng_key = rnd.PRNGKey(0)

    train_state = init_train_state(
        model=model, optimizer=optimizer, rng=rng_key)

    train_dataloader = DataLoader(
        config.data.train_dataset, batch_size=config.data.batch_size, num_workers=config.data.num_workers)

    if config.data.val_dataset is not None:
        val_dataloader = DataLoader(
            config.data.val_dataset, batch_size=config.data.batch_size, num_workers=config.data.num_workers)

    for epoch in tqdm(range(config.data.epochs)):
        for image, label in train_dataloader:
            image, label = jnp.asarray(image), jnp.asarray(label)
            rng_key, step_rng = rnd.split(rng_key)

            noise = rnd.normal(step_rng, label.shape)
            timesteps = rnd.choice(
                rng_key, num_timesteps, shape=(label.shape[0],))
            noised_seg_maps = scheduler.add_noise(
                scheduler_state, label, noise=noise, timesteps=timesteps
            )
            X = jnp.concatenate((image, noised_seg_maps), axis=1)

            train_state, loss = train_step(
                state=train_state, X=X, noise=noise, timesteps=timesteps
            )
            break
        break
    print("Hi")

    # wandb.log({'loss/train': loss})

    # if config.data.val_dataset is not None:
    #     for batch in val_dataloader:
    #         batch_array = jnp.asarray(batch)
    #         rng_key, step_rng = rnd.split(rng_key)

    #         noise = rnd.normal(step_rng, label.shape)
    #         timesteps = rnd.choice(
    #             rng_key, num_timesteps, shape=(label.shape[0],))
    #         X = jnp.concatenate((image, noise), axis=1)

    #         train_state, loss = eval_step(
    #             state=train_state, batch=batch_array)

    #         # wandb.log({'loss/val': loss})

    return train_state.params


def init_train_state(model: nn.Module, optimizer: optax.GradientTransformation, rng: KeyArray) -> train_state.TrainState:
    variables = model.init_weights(rng)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        tx=optimizer,
        params=variables,
    )


def train_step(
    state: train_state.TrainState, X: jax.Array, noise: jax.Array, timesteps: jax.Array,
):
    def loss_fn(params: jnp.array) -> Callable:
        predicted_noise = state.apply_fn(
            {'params': params}, X, timesteps=timesteps)
        loss = optax.l2_loss(predictions=predicted_noise, targets=noise)

        return loss.mean()

    gradient_fn = jax.value_and_grad(loss_fn)
    value, grads = gradient_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state, value


@jax.jit
def eval_step(state: train_state.TrainState, X: jax.Array, label: jax.Array):
    logits = state.apply_fn({'params': state.params}, image)
    rng, step_rng = rnd.split(state.rng)
    noise = rnd.normal(rng, label.shape)
    timesteps = rnd.choice(
        rng, state.timesteps, shape=(label.shape[0],))
    noised_seg_maps = state.scheduler.add_noise(
        state.scheduler_state, label, noise=noise, timesteps=timesteps)

    X = jnp.concatenate((image, noised_seg_maps), axis=1)

    def loss_fn(params: jnp.array) -> Callable:
        predicted_noise = state.apply_fn({'params': params}, X)
        loss = optax.l2_loss(predictions=predicted_noise, targets=noise)

        return loss

    state = state.replace(rng=rng)
    return state, loss_fn(logits, label)


@jax.jit
def inference(model: nn.Module, parameters: jax.Array, scheduler: FlaxSchedulerMixin, image: jnp.ndarray, shape: jax.Array, noise_generator, rng: KeyArray, debug: bool = False):
    rng, step_rng = rnd.split(rng)
    noisy_label = rnd.normal(step_rng, shape)
    scheduler_state = scheduler.create_state()

    def loop_func(step, data):
        noisy_label, scheduler_state = data
        X = jnp.concatenate((image, noisy_label), axis=1)

        noise = model.apply({'params': parameters}, X)
        latents, scheduler_state = scheduler.step(
            scheduler_state, noise, step, latents)

        return latents, scheduler_state

    if debug:
        for i in range(scheduler_state.num_inference_steps):
            noisy_label, scheduler_state = loop_func(
                i, (noisy_label, scheduler_state))
    else:
        sample, _ = jax.lax.fori_loop(
            0, scheduler_state.num_inference_steps, loop_func, (noisy_label, scheduler_state))

    return sample
