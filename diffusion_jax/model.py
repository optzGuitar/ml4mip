from typing import Optional, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from diffusers.models.unet_2d_blocks_flax import FlaxDownBlock2D, FlaxUpBlock2D
from diffusers.models.resnet_flax import FlaxResnetBlock2D
from diffusers.models.attention_flax import FlaxAttention
from diffusers.models.embeddings_flax import FlaxTimesteps, FlaxTimestepEmbedding

class UNetMidBlock2D(nn.Module):
    in_channels: int
    dropout: float = 0.0
    num_layers: int = 1
    add_attention: bool = True
    attention_head_dim: int = 64

    @nn.compact
    def __call__(self, hidden_states: jax.Array, temb: jax.Array, shape: jax.Array) -> jax.Array:
        hidden_states = FlaxResnetBlock2D(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            dropout_prob=self.dropout,
        )(hidden_states, temb)

        for _ in range(self.num_layers):
            if self.add_attention:
                hidden_states = FlaxAttention(
                    query_dim=self.in_channels,
                    heads=self.in_channels // self.attention_head_dim,
                    dim_head=self.attention_head_dim,
                    dropout=self.dropout,
                    split_head_dim=True,
                )(hidden_states).reshape(shape)
            else:
                hidden_states = FlaxResnetBlock2D(
                    in_channels=self.in_channels,
                    out_channels=self.in_channels,
                    dropout_prob=self.dropout,
                )(hidden_states, temb)

        return hidden_states


class FlaxUNet2DModel(nn.Module):
    r"""
    A 2D UNet model that takes a noisy sample, and a timestep and returns a sample
    shaped output.

    This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    subclass. Use it as a regular Flax Linen module and refer to the Flax documentation for all matters related to its
    general usage and behavior.

    Parameters:
        sample_size (`int`, *optional*):
            The size of the input sample.
        in_channels (`int`, *optional*, defaults to 4):
            The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4):
            The number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxDownBlock2D")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("FlaxUpBlock2D", "FlaxCrossAttnUpBlock2D", "FlaxCrossAttnUpBlock2D", "FlaxCrossAttnUpBlock2D")`):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        attention_head_dim (`int` or `Tuple[int]`, *optional*, defaults to 8):
            The dimension of the attention heads.
        num_attention_heads (`int` or `Tuple[int]`, *optional*):
            The number of attention heads.
        cross_attention_dim (`int`, *optional*, defaults to 768):
            The dimension of the cross attention features.
        dropout (`float`, *optional*, defaults to 0):
            Dropout probability for down, up and bottleneck blocks.
        flip_sin_to_cos (`bool`, *optional*, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, *optional*, defaults to 0): The frequency shift to apply to the time embedding.
        split_head_dim (`bool`, *optional*, defaults to `False`):
            Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
            enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
    """

    sample_size: int = 32
    in_channels: int = 4
    out_channels: int = 4
    num_blocks: int = 4

    block_out_channels: tuple[int, ...] = (320, 640, 1280, 1280)
    layers_per_block: int = 2
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32
    flip_sin_to_cos: bool = True
    freq_shift: int = 0
    split_head_dim: bool = False
    transformer_layers_per_block: Union[int, tuple[int, ...]] = 1
    addition_embed_type: Optional[str] = None
    addition_time_embed_dim: Optional[int] = None
    addition_embed_type_num_heads: int = 64
    projection_class_embeddings_input_dim: Optional[int] = None

    def init_weights(self, rng: jax.Array) -> FrozenDict:
        # init input tensors
        sample_shape = (1, self.in_channels,
                        self.sample_size, self.sample_size)
        sample = jnp.zeros(sample_shape, dtype=jnp.float32)
        timesteps = jnp.ones((1,), dtype=jnp.int32)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.init(rngs, sample, timesteps)["params"]

    def setup(self) -> None:
        block_out_channels = self.block_out_channels
        time_embed_dim = block_out_channels[0] * 4

        # input
        self.conv_in = nn.Conv(
            block_out_channels[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        # time
        self.time_proj = FlaxTimesteps(
            block_out_channels[0], flip_sin_to_cos=self.flip_sin_to_cos, freq_shift=self.freq_shift
        )
        self.time_embedding = FlaxTimestepEmbedding(
            time_embed_dim, dtype=self.dtype)

        # transformer layers per block
        transformer_layers_per_block = self.transformer_layers_per_block
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [
                transformer_layers_per_block] * self.num_blocks

        # addition embed types
        if self.addition_embed_type is None:
            self.add_embedding = None
        elif self.addition_embed_type == "text_time":
            if self.addition_time_embed_dim is None:
                raise ValueError(
                    f"addition_embed_type {self.addition_embed_type} requires `addition_time_embed_dim` to not be None"
                )
            self.add_time_proj = FlaxTimesteps(
                self.addition_time_embed_dim, self.flip_sin_to_cos, self.freq_shift)
            self.add_embedding = FlaxTimestepEmbedding(
                time_embed_dim, dtype=self.dtype)
        else:
            raise ValueError(
                f"addition_embed_type: {self.addition_embed_type} must be None or `text_time`.")

        # down
        down_blocks = []
        output_channel = block_out_channels[0]
        for i in range(self.num_blocks):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = FlaxDownBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                dropout=self.dropout,
                num_layers=self.layers_per_block,
                add_downsample=not is_final_block,
                dtype=self.dtype,
            )

            down_blocks.append(down_block)
        self.down_blocks = down_blocks

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            dropout=self.dropout,
            num_layers=self.layers_per_block,
        )

        # up
        up_blocks = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(self.num_blocks):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(
                i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = FlaxUpBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                num_layers=self.layers_per_block + 1,
                add_upsample=not is_final_block,
                dropout=self.dropout,
                dtype=self.dtype,
            )

            up_blocks.append(up_block)
            prev_output_channel = output_channel
        self.up_blocks = up_blocks

        # out
        self.conv_norm_out = nn.GroupNorm(num_groups=32, epsilon=1e-5)
        self.conv_out = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(
        self,
        sample: jnp.ndarray,
        timesteps: Union[jnp.ndarray, float, int],
        return_dict: bool = True,
        train: bool = False,
    ) -> jnp.ndarray:
        r"""
        Args:
            sample (`jnp.ndarray`): (batch, channel, height, width) noisy inputs tensor
            timestep (`jnp.ndarray` or `float` or `int`): timesteps
            encoder_hidden_states (`jnp.ndarray`): (batch_size, sequence_length, hidden_size) encoder hidden states
            added_cond_kwargs: (`dict`, *optional*):
                A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
                are passed along to the UNet blocks.
            down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
                A tuple of tensors that if specified are added to the residuals of down unet blocks.
            mid_block_additional_residual: (`torch.Tensor`, *optional*):
                A tensor that if specified is added to the residual of the middle unet block.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] instead of a
                plain tuple.
            train (`bool`, *optional*, defaults to `False`):
                Use deterministic functions and disable dropout when not training.

        Returns:
            [`~models.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`.
            When returning a tuple, the first element is the sample tensor.
        """
        # 1. time
        if not isinstance(timesteps, jnp.ndarray):
            timesteps = jnp.array([timesteps], dtype=jnp.int32)
        elif isinstance(timesteps, jnp.ndarray) and len(timesteps.shape) == 0:
            timesteps = timesteps.astype(dtype=jnp.float32)
            timesteps = jnp.expand_dims(timesteps, 0)

        t_emb = self.time_proj(timesteps)
        t_emb = self.time_embedding(t_emb)

        # 2. pre-process
        sample = jnp.transpose(sample, (0, 2, 3, 1))
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for down_block in self.down_blocks:
            sample, res_samples = down_block(
                sample, t_emb, deterministic=not train)
            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(
            sample, t_emb, shape=sample.shape)

        # 5. up
        for up_block in self.up_blocks:
            res_samples = down_block_res_samples[-(self.layers_per_block + 1):]
            down_block_res_samples = down_block_res_samples[: -(
                self.layers_per_block + 1)]
            sample = up_block(
                sample, temb=t_emb, res_hidden_states_tuple=res_samples, deterministic=not train)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = nn.silu(sample)
        sample = self.conv_out(sample)
        sample = jnp.transpose(sample, (0, 3, 1, 2))

        if not return_dict:
            return (sample,)

        return sample
