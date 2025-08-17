from typing import cast, Optional, Tuple, Union
import torch
import captum.attr

from ..modules.DeepTaylor import DTD


from diffusers.models.unets import UNet2DConditionModel, CrossAttnDownBlock2D

def test_dtd_root():
    torch.manual_seed(0)
    net = CrossAttnDownBlock2D(32, 640, 32)

    idx = 0
    x = (0.25 * torch.randn(2, 3, requires_grad=True) + 1).clamp(min=0)

    rules: list[DTD.RULE] = ["0", "z+", "w2", DTD.GammaRule(1000)]
    for rule in rules:
        x_root = DTD.calculate_root_for_single_neuron(
            x, net.layer1, idx, rule=rule
        )

        assert x_root is not None

        x_root.shape, x.shape
        print("x", x)
        print("x_root", x_root)
        print("out", net.layer1.linear(x)[:, idx].tolist())
        print("out root", net.layer1.linear(x_root)[:, idx].tolist())

        root_output = net.layer1.linear(x_root)[:, idx]
        assert torch.allclose(
            root_output, torch.zeros_like(root_output), atol=1e-6
        )

test_dtd_root()
    