import torch
import torch.nn as nn

from .convbnrelu import ConvBNRelu
from .noise import Noise


class Encoder(nn.Module):

    def __init__(self, msg_length, channels=64, num_blocks=4, last_tanh=True):
        super(Encoder, self).__init__()
        layers = [ConvBNRelu(3, channels)]

        for _ in range(num_blocks - 1):
            layer = ConvBNRelu(channels, channels)
            layers.append(layer)

        self.conv_bns = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(
            channels + 3 + msg_length, channels)

        self.final_layer = nn.Conv2d(channels, 3, kernel_size=1)

        self.tanh = nn.Tanh() if last_tanh else None

    def forward(self, imgs, msgs):
        N, C, H, W = imgs.shape

        msgs = msgs.unsqueeze(-1).unsqueeze(-1)
        msgs = msgs.repeat(1, 1, H, W)

        encoded_imgs = self.conv_bns(imgs)

        concat = torch.cat([msgs, encoded_imgs, imgs], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)

        if self.tanh:
            im_w = self.tanh(im_w)

        return im_w


class Decoder(nn.Module):

    def __init__(self, msg_length, channels=64, num_blocks=8):
        super(Decoder, self).__init__()

        layers = [ConvBNRelu(3, channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBNRelu(channels, channels))

        layers.append(ConvBNRelu(channels, msg_length))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(msg_length, msg_length)

    def forward(self, img_w):
        x = self.layers(img_w)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear(x)

        return x


class EncoderDecoder(nn.Module):
    def __init__(self, msg_length, encode_weight, noises) -> None:
        super(EncoderDecoder, self).__init__()

        self.msg_length = msg_length

        self.encode_weight = encode_weight

        self.encoder = Encoder(msg_length)
        self.decoder = Decoder(msg_length)

        self.noise = Noise(noises)

    def forward(self, imgs, msgs):

        watermark = self.encoder(imgs, msgs)

        encoded_imgs = imgs + self.encode_weight * watermark

        noised_imgs = self.noise(encoded_imgs)

        decoded_msgs = self.decoder(noised_imgs)

        return decoded_msgs, encoded_imgs, noised_imgs
