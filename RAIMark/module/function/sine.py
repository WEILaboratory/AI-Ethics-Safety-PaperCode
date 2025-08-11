import numpy as np
import torch
import torch.nn as nn


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30,
                              np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class Sine(nn.Module):
    def __init__(self):
        super(Sine, self).__init__()

    def forward(self, input):
        return torch.sin(30 * input)
