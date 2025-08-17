import os
import copy
import json
import math

import tarfile
import tempfile
import shutil
from typing import Optional, Tuple, Union

import torch
from torch import nn
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from diffusers.utils import logging, deprecate, is_torch_version
from .modules.resnet import Downsample2D, ResnetBlock2D, ResnetBlockCondNorm2D

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'stable-diffusion-xlv1.0': "/data/junlei/project-code/diffusion/models/stable-diffusion-xl-base-1.0",
}

CONFIG_NAME = {
    'stable-diffusion-xlv1.0': 'sdxl_config.json',
}


class DiffusionUnetConfig(object):

    def __init__(self,
        diffusion_unet_config_json_file,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: float = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads: int = 64,
        ):
        if isinstance(diffusion_unet_config_json_file, str):
            with open(diffusion_unet_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(diffusion_unet_config_json_file, int):
            self.sample_size = sample_size
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.center_input_sample = center_input_sample
            self.flip_sin_to_cos = flip_sin_to_cos
            self.freq_shift = freq_shift
            self.down_block_types = down_block_types
            self.mid_block_type = mid_block_type
            self.up_block_types = up_block_types
            self.only_cross_attention = only_cross_attention
            self.block_out_channels = block_out_channels
            self.layers_per_block = layers_per_block
            self.downsample_padding = downsample_padding
            self.mid_block_scale_factor = mid_block_scale_factor
            self.dropout = dropout
            self.act_fn = act_fn
            self.norm_num_groups = norm_num_groups
            self.norm_eps = norm_eps
            self.cross_attention_dim = cross_attention_dim
            self.transformer_layers_per_block = transformer_layers_per_block
            self.reverse_transformer_layers_per_block = reverse_transformer_layers_per_block
            self.encoder_hid_dim = encoder_hid_dim
            self.encoder_hid_dim_type = encoder_hid_dim_type
            self.attention_head_dim = attention_head_dim
            self.num_attention_heads = num_attention_heads
            self.dual_cross_attention = dual_cross_attention
            self.use_linear_projection = use_linear_projection
            self.class_embed_type = class_embed_type
            self.addition_embed_type = addition_embed_type
            self.addition_time_embed_dim = addition_time_embed_dim
            self.num_class_embeds = num_class_embeds
            self.upcast_attention = upcast_attention
            self.resnet_time_scale_shift = resnet_time_scale_shift
            self.resnet_skip_time_act = resnet_skip_time_act
            self.resnet_out_scale_factor = resnet_out_scale_factor
            self.time_embedding_type = time_embedding_type
            self.time_embedding_dim = time_embedding_dim
            self.time_embedding_act_fn = time_embedding_act_fn
            self.timestep_post_act = timestep_post_act
            self.time_cond_proj_dim = time_cond_proj_dim
            self.conv_in_kernel = conv_in_kernel
            self.conv_out_kernel = conv_out_kernel
            self.projection_class_embeddings_input_dim = projection_class_embeddings_input_dim
            self.attention_type = attention_type
            self.class_embeddings_concat = class_embeddings_concat
            self.mid_block_only_cross_attention = mid_block_only_cross_attention
            self.cross_attention_norm = cross_attention_norm
            self.addition_embed_type_num_heads = addition_embed_type_num_heads
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                            "or the path to a pretrained model config file (str)")


    @classmethod
    def from_dict(cls, json_object):
        config = DiffusionUnetConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config
    

class UnetDownBlock2D(nn.Module):

    def __init__(
            self,         
            in_channels: int,
            out_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            output_scale_factor: float = 1.0,
            add_downsample: bool = True,
            downsample_padding: int = 1,
        ):
        super(UnetDownBlock2D, self).__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None, *args, **kwargs
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        output_states = ()

        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb
                    )
            else:
                hidden_states = resnet(hidden_states, temb)

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states
    
