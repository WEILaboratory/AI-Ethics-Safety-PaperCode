import torch
import torch.nn as nn


class RegressionModel(torch.nn.Module):
    def __init__(self, n_feats, n_hidden):
        super(RegressionModel, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(n_feats, 1))

    def forward(self, x):
        y = x
        for i in range(len(self.layers)):
            y_temp = self.layers[i](y)
            if i < len(self.layers) - 1:
                y = torch.tanh(y_temp)
            else:
                y = y_temp
        return y
