import torch.nn as nn
import random

from .noise_layers import *

class Noise(nn.Module):
    def __init__(self, noises):
        super(Noise, self).__init__()

        self.noises = [eval(noise) for noise in noises]

        for noise in self.noises:
            noise.eval()

    def sample(self):
        self.noise = random.choice(self.noises)

    def forward(self, image_and_cover):        
        return self.noise(image_and_cover)
