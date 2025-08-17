"""Implementation of the LRP_a1_b0 rules for the Relation Network."""

from copy import deepcopy
from typing import Callable, Optional, Union, cast

import captum.attr
import numpy as np
import torch
from captum.attr._utils import lrp_rules
from torch import nn

import utils
from typing import Any, Callable, Generic, Optional, Sequence, TypeVar, Union, Tuple, Dict
from diffusers.utils import deprecate, is_torch_version, logging
from modules.unets import UNet2DConditionModel, UNet2DConditionOutput, CrossAttnDownBlock2D, CrossAttnUpBlock2D

logger = logging.get_logger(__name__)  #  pylint: disable=invalid-name

class Sum(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.sum(x, dim=self.dim)


class Concat(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)


class ConcatRule(lrp_rules.EpsilonRule):
    def __init__(self, dim: int = 0) -> None:
        super().__init__(epsilon=1e-9)
        self.dim = dim

    def forward_hook(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        outputs: torch.Tensor,
    ) -> torch.Tensor:
        def _create_backward_hook_input(
            input: torch.Tensor, start: int, end: int
        ) -> Callable[[torch.Tensor], None]:
            def _backward_hook_input(grad: torch.Tensor) -> None:
                rel = self.relevance_output[grad.device]  # type: ignore
                idx = [slice(None, None, None) for _ in range(grad.dim())]
                idx[self.dim] = slice(start, end)
                return rel[idx]

            return _backward_hook_input

        """Register backward hooks on input and output
        tensors of linear layers in the model."""
        inputs = lrp_rules._format_tensor_into_tuples(inputs)
        self._has_single_input = len(inputs) == 1
        self._handle_input_hooks = []
        offset = 0
        for input in inputs:
            if not hasattr(input, "hook_registered"):
                next_offset = offset + input.size(self.dim)
                input_hook = _create_backward_hook_input(
                    input.data, offset, next_offset
                )
                offset = next_offset
                self._handle_input_hooks.append(input.register_hook(input_hook))
                input.hook_registered = True  # type: ignore
        output_hook = self._create_backward_hook_output(outputs.data)
        self._handle_output_hook = outputs.register_hook(output_hook)
        return outputs.clone()


class LRPViewOfRelationNetwork(nn.Module):
    def __init__(self, relnet: CrossAttnDownBlock2D):
        super().__init__()

        # do not include in the modules
        self._relnet = [relnet]
        self.resnets = deepcopy(relnet.resnets)
        self.attentions = deepcopy(relnet.attentions)
        if relnet.downsample:
            self.downsample = deepcopy(relnet.downsample)

        self.concat = Concat(dim=2)
        self.sum = Sum(dim=1)

    @property
    def original_relnet(self) -> CrossAttnDownBlock2D:
        return self._relnet[0]

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        additional_residuals: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        output_states = ()

        blocks = list(zip(self.resnets, self.attentions))

        for i, (resnet, attn) in enumerate(blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

            # apply additional residuals to the output of the last pair of resnet and attention blocks
            if i == len(blocks) - 1 and additional_residuals is not None:
                hidden_states = hidden_states + additional_residuals

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states

    def get_lrp_saliency(
        self,
        image: torch.Tensor,
        question: torch.Tensor,
        q_len: int,
        target: Union[int, torch.Tensor],
        question_permutation: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        return self.get_lrp_saliency_and_logits(
            image,
            question,
            q_len,
            target,
            question_permutation,
            normalize,
        )[0]

    def get_lrp_saliency_and_logits(
        self,
        image: torch.Tensor,
        question: torch.Tensor,
        q_len: int,
        bias: Union[int, torch.Tensor],
        question_permutation: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        set_lrp_rules(self)
        lrp_attr = captum.attr.LRP(self)

        if isinstance(bias, int):
            bias = image.new_full((image.size(0),), bias, dtype=torch.long)

        bias_embed = self.original_relnet.attentions(32, 640, 32)


        if question_permutation is not None:
            shape = bias_embed.shape
            bias_embed = bias_embed[
                :, question_permutation
            ].contiguous()
            assert bias_embed.shape == shape

        saliency = lrp_attr.attribute(
            image,
            target=bias,
            additional_forward_args=(bias_embed,),
            verbose=False,
        )

        logits = self(image, bias_embed)

        if normalize:
            explained_logit = logits[torch.arange(len(logits)), bias]
            # to make the heatmaps comparable we normalize the saliencies
            saliency = saliency / explained_logit[:, None, None, None]

        return saliency, logits


def set_lrp_rules(lrp_relnet: nn.Module, set_bias_to_zero: bool = True) -> None:
    for module in lrp_relnet.modules():
        if isinstance(module, (nn.Conv2d, nn.ReLU, nn.Linear, Sum)):
            module.rule = lrp_rules.Alpha1_Beta0_Rule(
                set_bias_to_zero=set_bias_to_zero
            )
        elif isinstance(module, Concat):
            module.rule = ConcatRule(module.dim)
        else:
            module.rule = lrp_rules.IdentityRule()


def normalize_saliency(
    saliency: torch.Tensor,
    clip_percentile_max: Optional[float] = 99.5,
    clip_percentile_min: Optional[float] = 0.5,
    retain_zero: bool = False,
    abs: bool = True,
) -> torch.Tensor:
    assert not retain_zero
    if abs:
        saliency = saliency.abs()
    saliency_np = utils.to_np(saliency)
    vmin, vmax = np.percentile(
        saliency_np, [clip_percentile_min or 0, clip_percentile_max or 100]
    )
    saliency = saliency.clamp(vmin, vmax)
    return (saliency - vmin) / (vmax - vmin)