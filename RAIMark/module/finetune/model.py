import copy
import random
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from .noise_layers import *


class Model(nn.Module):
    def __init__(self, funcimg, noise, decoder, msg_length):
        super(Model, self).__init__()
        
        self.funcimg = funcimg

        self.noise = [eval(i) for i in noise]

        self.train_noise = None

        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.unnormalize_img = transforms.Normalize(mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5], std=[1/0.5, 1/0.5, 1/0.5])

        self.decoder = decoder

        self.msg_length = msg_length

        self.size = (256, 256)

    def forward(self, sampler, eval=False):
        sampled_img = sampler(self.funcimg)

        unnormolized_img = self.unnormalize_img(sampled_img)

        if eval:
            noised_imgs = []
            decoded_msgs = []

            for noise in self.noise:
                noised_img = noise(unnormolized_img)
                noised_imgs.append(noised_img)

                to_decode_img = F.interpolate(noised_img, size=self.size, mode="bilinear", align_corners=False)
                to_decode_img = self.normalize_img(to_decode_img)

                decoded_msgs.append(self.decoder(to_decode_img)[:, :self.msg_length])

            return decoded_msgs, sampled_img, noised_imgs, self.noise
        
        else:
            if not self.train_noise:
                self.train_noise = copy.copy(self.noise)

                random.shuffle(self.train_noise)

            noise = self.train_noise.pop()

            noised_img = noise(unnormolized_img)

            to_decode_img = F.interpolate(noised_img, size=self.size, mode="bilinear", align_corners=False)
            to_decode_img = self.normalize_img(to_decode_img)

            decoded_msg = self.decoder(to_decode_img)

            return decoded_msg[:, :self.msg_length], sampled_img, noised_img, noise
