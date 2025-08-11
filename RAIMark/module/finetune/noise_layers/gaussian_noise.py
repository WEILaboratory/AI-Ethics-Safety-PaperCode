import numpy as np
import torch
import torch.nn as nn


class GN(nn.Module):

    def __init__(self, std, mean=0):
        super(GN, self).__init__()
        self.std = std
        self.mean = mean

    def gaussian_noise(self, image, mean, std):
        noise = torch.Tensor(np.random.normal(mean, std / 2, image.shape)).to(image.device)
        out = image + noise
        return out

    def forward(self, image):
        return self.gaussian_noise(image, self.mean, self.std)
    
    def __repr__(self) -> str:
        return f'GN({self.std})'
    


