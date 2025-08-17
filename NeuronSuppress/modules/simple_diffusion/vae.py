import torch
from torch import nn
from collections import OrderedDict

class Resnet(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.ResnetModel = nn.Sequential(OrderedDict([
            ('groupnorm1', nn.GroupNorm(num_groups=32, num_channels=dim_in, eps=1e-6, affine=True)),
            ('silu1', nn.SiLU()),
            ('conv1', nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1)),
            ('groupnorm2', nn.GroupNorm(num_groups=32, num_channels=dim_out, eps=1e-6, affine=True)),
            ('silu2', nn.SiLU()),
            ('conv2', nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1))
        ]))

        self.residual = None

        if dim_in != dim_out:
            self.residual = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        
        residual = x
        if self.residual:
            residual = self.residual(x)
        return residual + self.ResnetModel(x)
    
    
# print(Resnet(128, 256)(torch.randn(1, 128, 10, 10)).shape)
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.norm = nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6, affine=True)

        self.q = nn.Linear(512, 512)
        self.k = nn.Linear(512, 512)
        self.v = nn.Linear(512, 512)
        self.out = nn.Linear(512, 512)
    
    def forward(self, x):
        res = x
        
        x = self.norm(x)
        x = x.flatten(start_dim=2).transpose(1,2)

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        k = k.transpose(1, 2)

        atten = torch.baddbmm(torch.empty(1, 4096, 4096, device=q.device), q, k, beta=0, alpha=0.044194173824159216)
        atten = torch.softmax(atten, dim=2)

        atten = atten.bmm(v)

        atten = self.out(atten)
        atten = atten.transpose(1, 2).reshape(-1, 512, 64, 64)
        atten = atten + res
        return atten
# print(Attention()(torch.randn(1, 512, 64, 64)).shape)

class Pad(nn.Module):
    def forward(self, x):
        return nn.functional.pad(x, (0, 1, 0, 1), mode='constant', value=0)
    
# print(Pad()(torch.ones(1, 2, 5, 5)))

class EncoderBlock(nn.Module):
    def __init__(self):
        super(EncoderBlock, self).__init__()

        self.EncoderBlock = nn.Sequential(OrderedDict([
            #  in
            ('conv1', nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)),
            #  down
            ('down_encoder_block1', nn.Sequential(
                Resnet(128, 128),
                Resnet(128, 128),
                nn.Sequential(
                    Pad(),
                    nn.Conv2d(128, 128, 3, stride=2, padding=0),
                ),
            )),

            ('down_encoder_block2', nn.Sequential(
                Resnet(128, 256),
                Resnet(256, 256),
                nn.Sequential(
                    Pad(),
                    nn.Conv2d(256, 256, 3, stride=2, padding=0),
                ),
            )),

            ('down_encoder_block3', nn.Sequential(
                Resnet(256, 512),
                Resnet(512, 512),
                nn.Sequential(
                    Pad(),
                    nn.Conv2d(512, 512, 3, stride=2, padding=0),
                ),
            )),

            ('down_encoder_block4', nn.Sequential(
                Resnet(512, 512),
                Resnet(512, 512),
            )),

            #  mid
            ('mid_encoder_block', nn.Sequential(
                Resnet(512, 512),
                Attention(),
                Resnet(512, 512),
            )),

            #  out
            ('out_encoder_block', nn.Sequential(
                nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6),
                nn.SiLU(),
                nn.Conv2d(512, 8, 3, padding=1),
            )),
            
            #  norm distribution layer
            ('norm_layer', nn.Conv2d(8, 8, 1)),
        ]))


    def forward(self, x):
        x = self.EncoderBlock(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self):
        super(DecoderBlock, self).__init__()
        
        self.DecoderBlock = nn.Sequential(OrderedDict([
            #  norm distribution layer
            ('norm_layer', nn.Conv2d(4, 4, 1)),

            #  in 
            ('conv1', nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1)),

            #  middle
            ('mid_decoder_block', nn.Sequential(
                Resnet(512, 512), 
                Attention(),
                Resnet(512, 512),
            )),

            #  up
            ('up_decoder_block1', nn.Sequential(
                Resnet(512, 512),
                Resnet(512, 512),
                Resnet(512, 512),
                nn.Upsample(scale_factor=2.0, mode='nearest'),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
            )),
            ('up_decoder_block2', nn.Sequential(
                Resnet(512, 512),
                Resnet(512, 512),
                Resnet(512, 512),
                nn.Upsample(scale_factor=2.0, mode='nearest'),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
            )),
            ('up_decoder_block3', nn.Sequential(
                Resnet(512, 256),
                Resnet(256, 256),
                Resnet(256, 256),
                nn.Upsample(scale_factor=2.0, mode='nearest'),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
            )),
            ('up_decoder_block4', nn.Sequential(
                Resnet(256, 128),
                Resnet(128, 128),
                Resnet(128, 128),
            )),

            #  out
            ('out_decoder_block', nn.Sequential(
                nn.GroupNorm(num_channels=128, num_groups=32, eps=1e-6),
                nn.SiLU(),
                nn.Conv2d(128, 3, 3, padding=1),
            )),
        ]))


    def forward(self, x):
        x = self.DecoderBlock(x)
        return x
        
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = EncoderBlock().EncoderBlock
        self.decoder = DecoderBlock().DecoderBlock

    
    def sample(self, h):
        mean = h[:, :4]
        logvar = h[:, 4:]
        std = logvar.exp()**0.5

        h = torch.randn(mean.shape, device=mean.device)
        h = mean + std * h
        return h

    def forward(self, x):
        h = self.encoder(x)
        h = self.sample(h)
        h = self.decoder(h)
        return h

print(VAE()(torch.randn(1, 3, 512, 512)).shape)