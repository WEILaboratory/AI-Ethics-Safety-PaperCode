import torch
import torch.nn as nn
from .sine import *


def genlayers(in_channels, out_channels, hidden_layers, hidden_features):
    layers = []

    layers.append(nn.Linear(in_channels, hidden_features))
    layers.append(Sine())

    for i in range(hidden_layers):
        layers.append(nn.Linear(hidden_features, hidden_features))
        layers.append(Sine())

    layers.append(nn.Linear(hidden_features, out_channels))

    return nn.Sequential(*layers)


class FuncImg(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, hidden_layers=3, hidden_features=256):
        super(FuncImg, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if out_channels == 1:
            self.l = genlayers(in_channels, 1, hidden_layers, hidden_features)

            self.l.apply(sine_init)
            self.l[0].apply(first_layer_sine_init)

        if out_channels == 3:
            self.r = genlayers(in_channels, 1, hidden_layers, hidden_features)
            self.g = genlayers(in_channels, 1, hidden_layers, hidden_features)
            self.b = genlayers(in_channels, 1, hidden_layers, hidden_features)

            for m in [self.r, self.g, self.b]:
                m.apply(sine_init)
                m[0].apply(first_layer_sine_init)

    def forward(self, input):
        if self.out_channels == 1:
            return self.l(input)

        if self.out_channels == 3:
            r = self.r(input)
            g = self.g(input)
            b = self.b(input)

            return torch.cat([r, g, b], dim=1)
