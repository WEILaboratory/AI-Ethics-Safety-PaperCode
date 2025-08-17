import torch
from torch import nn
from collections import OrderedDict

class Resnet(nn.Module):
    
    def __init__(self, dim_in, dim_out):
        super(Resnet, self).__init__()

        # Time embedding layer
        self.time_embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1280, dim_out),
            nn.Unflatten(dim=1, unflattened_size=(dim_out, 1, 1)),
        )

        # First convolutional block
        self.conv_block1 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=dim_in, eps=1e-05, affine=True),
            nn.SiLU(),
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
        )

        # Second convolutional block
        self.conv_block2 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=dim_out, eps=1e-05, affine=True),
            nn.SiLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
        )

        # Residual connection
        self.residual = None
        if dim_in != dim_out:
            self.residual = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x, time):
        # Save the input tensor for the residual connection
        residual = x

        # Apply time embedding
        time_emb = self.time_embedding(time)

        # Apply the first convolutional block and add time embedding
        x = self.conv_block1(x) + time_emb

        # Apply the second convolutional block
        x = self.conv_block2(x)

        # Apply residual connection if dimensions do not match
        if self.residual:
            residual = self.residual(residual)

        # Add the residual connection
        x = x + residual

        return x
    
# print(Resnet(320, 640)(torch.randn(1, 320, 32, 32), torch.randn(1, 1280)).shape)


class CrossAttention(nn.Module):

    def __init__(self, dim_q, dim_kv):
        super(CrossAttention, self).__init__()

        self.dim_q = dim_q

        self.q = nn.Linear(dim_q, dim_q, bias=False)
        self.k = nn.Linear(dim_kv, dim_q, bias=False)
        self.v = nn.Linear(dim_kv, dim_q, bias=False)
        self.out = nn.Linear(dim_q, dim_q)
    
    def reshape(self, x, split_size):
        b, lens, dim = x.shape
        x = x.reshape(b, lens, split_size, dim // split_size)
        x = x.transpose(1, 2)
        x = x.reshape(b * split_size, lens, dim // split_size)
        return x
    
    def reshape_back(self, x, split_size):
        b, lens, dim = x.shape
        x = x.reshape(b // split_size, split_size, lens, dim)
        x = x.transpose(1, 2)
        x = x.reshape(b // split_size, lens, dim * split_size)
        return x
    
    def forward(self, q, kv):
        q = self.q(q)
        k = self.k(kv)
        v = self.v(kv)

        q = self.reshape(q, 8)
        k = self.reshape(k, 8)
        v = self.reshape(v, 8)

        scale = (self.dim_q // 8) ** -0.5
        atten = torch.baddbmm(
            torch.empty(q.shape[0], q.shape[1], k.shape[1], device=q.device),
            q, k.transpose(1, 2),
            beta=0,
            alpha=scale
        )

        atten = atten.softmax(dim=-1)
        atten = atten.bmm(v)

        atten = self.reshape_back(atten, 8)
        atten = self.out(atten)

        return atten
    
# print(CrossAttention(320, 768)(torch.randn(1, 4096, 320), torch.randn(1, 77, 768)).shape)

class Transformer(nn.Module):
    def __init__(self, dim):
        super(Transformer, self).__init__()

        self.dim = dim

        self.norm_in =  nn.GroupNorm(num_groups=32, num_channels=dim, eps=1e-6, affine=True)
        self.cnn_in = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        
        #  attention
        self.norm_atten0 = nn.LayerNorm(dim, elementwise_affine=True)
        self.cross_atten1 = CrossAttention(dim, dim)
        self.norm_atten1 = nn.LayerNorm(dim, elementwise_affine=True)
        self.cross_atten2 = CrossAttention(dim, 768)

        #  activation
        self.norm_act = nn.LayerNorm(dim, elementwise_affine=True)
        self.fc0 = nn.Linear(dim, dim * 8)
        self.act = nn.GELU()
        self.fc1 = nn.Linear(dim * 4, dim)

        self.cnn_out =  nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)


    def forward(self, q, kv):
        b, _, h, w = q.shape
        res1 = q

        # ---- Input Processing ----
        q = self.cnn_in(self.norm_in(q))
        q = q.permute(0, 2, 3, 1).reshape(b, h * w, self.dim)

        # ---- Attention ----
        q = self.cross_atten1(q=self.norm_atten0(q), kv=self.norm_atten0(q)) + q
        q = self.cross_atten2(q=self.norm_atten1(q), kv=kv) + q

        # ---- Activation ----
        res2 = q
        q = self.fc0(self.norm_act(q))

        d = q.shape[2] // 2
        q = q[:, :, :d] * self.act(q[:, :, d:])
        q = self.fc1(q) + res2

        # ---- Output Processing ----
        q = q.reshape(b, h, w, self.dim).permute(0, 3, 1, 2).contiguous()
        q = self.cnn_out(q) + res1

        return q

# print(Transformer(320)(torch.randn(1, 320, 64, 64), torch.randn(1, 77, 768)).shape)

class DownBlock(nn.Module):
    
    def __init__(self, dim_in, dim_out) -> None:
        super(DownBlock, self).__init__()

        self.transformer1 = Transformer(dim_out)
        self.resnet1 = Resnet(dim_in, dim_out)

        self.transformer2 = Transformer(dim_out)
        self.resnet2 = Resnet(dim_out, dim_out)

        self.downsample = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=2, padding=1)

    def forward(self, out_vae, out_encoder, time):
        outs = []

        out_vae = self.resnet1(out_vae, time)
        out_vae = self.transformer1(out_vae, out_encoder)
        outs.append(out_vae)

        out_vae = self.resnet2(out_vae, time)
        out_vae = self.transformer2(out_vae, out_encoder)
        outs.append(out_vae)

        out_vae = self.downsample(out_vae)
        outs.append(out_vae)

        return out_vae, outs

# print(DownBlock(320, 640)(torch.randn(1, 320, 32, 32), torch.randn(1, 77, 768), torch.randn(1, 1280))[0].shape)

class UpBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_prev, add_up):
        super(UpBlock, self).__init__()

        self.resnet1 = Resnet(dim_out + dim_prev, dim_out)
        self.resnet2 = Resnet(dim_out + dim_out, dim_out)
        self.resnet3 = Resnet(dim_in + dim_out, dim_out)

        self.transformer1 = Transformer(dim_out)
        self.transformer2 = Transformer(dim_out)
        self.transformer3 = Transformer(dim_out)

        self.upsample = None
        if add_up:
            self.upsample = nn.Sequential(OrderedDict([
                ('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
                ('conv', nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1)),
            ]))

    def forward(self, out_vae, out_encoder, time, out_down):
        out_vae = self.resnet1(torch.cat([out_vae, out_down.pop()], dim=1), time)
        out_vae = self.transformer1(out_vae, out_encoder)

        out_vae = self.resnet2(torch.cat([out_vae, out_down.pop()], dim=1), time)
        out_vae = self.transformer2(out_vae, out_encoder)

        out_vae = self.resnet3(torch.cat([out_vae, out_down.pop()], dim=1), time)
        out_vae = self.transformer3(out_vae, out_encoder)

        if self.upsample:
            out_vae = self.upsample(out_vae)
        return out_vae
# print(UpBlock(320, 640, 1280, True)(torch.randn(1, 1280, 32, 32),
#                                     torch.randn(1, 77, 768),
#                                     torch.randn(1, 1280),
#                                     [torch.randn(1, 320, 32, 32), 
#                                     torch.randn(1, 640, 32, 32),
#                                     torch.randn(1, 640, 32, 32)]).shape)    

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        #  input layer
        self.in_vae = nn.Conv2d(4, 320, kernel_size=3, padding=1)
        self.in_time = nn.Sequential(
            nn.Linear(320, 1280),
            nn.SiLU(),
            nn.Linear(1280, 1280),
        )

        #  Down_sampling blocks
        self.down_block1 = DownBlock(320, 320)
        self.down_block2 = DownBlock(320 ,640)
        self.down_block3 = DownBlock(640, 1280)

        self.down_resnet1 = Resnet(1280, 1280)
        self.down_resnet2 = Resnet(1280, 1280)

        #  Mid layers
        self.mid_resnet1 = Resnet(1280, 1280)
        self.mid_transformer = Transformer(1280)
        self.mid_resnet2 = Resnet(1280, 1280)

        #  Upsampling layers
        self.up_resnet1 = Resnet(2560, 1280)
        self.up_resnet2 = Resnet(2560, 1280)
        self.up_resnet3 = Resnet(2560, 1280)

        self.up_in = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(1280, 1280, kernel_size=3, padding=1),
        )

        #  Upsampling blocks
        self.up_block1 = UpBlock(640, 1280, 1280, True)
        self.up_block2 = UpBlock(320, 640, 1280, True)
        self.up_block3 = UpBlock(320, 320, 640, False)

        #  Output layer
        self.out = nn.Sequential(
            nn.GroupNorm(num_channels=320, num_groups=32, eps=1e-5),
            nn.SiLU(),
            nn.Conv2d(320, 4, kernel_size=3, padding=1),
        )

    def get_time_embed(self, t):
        e = torch.arange(160) * -9.210340371976184 / 160    #   -9.210340371976184 = -math.log(10000)
        e = e.exp().to(t.device) * t
        e = torch.cat([e.cos(), e.sin()]).unsqueeze(dim=0)
        return e
    
    def forward(self, out_vae, out_encoder, time):
        #  Input processing
        out_vae = self.in_vae(out_vae)
        time = self.get_time_embed(time)
        time = self.in_time(time)

        #  Downsampling
        out_down = [out_vae]
        out_vae, out = self.down_block1(out_vae=out_vae, out_encoder=out_encoder, time=time)
        out_down.extend(out)
        out_vae, out = self.down_block2(out_vae=out_vae, out_encoder=out_encoder, time=time)
        out_down.extend(out)
        out_vae, out = self.down_block3(out_vae=out_vae, out_encoder=out_encoder, time=time)
        out_down.extend(out)
    
        out_vae = self.down_resnet1(out_vae, time)
        out_down.append(out_vae)
        out_vae = self.down_resnet2(out_vae, time)
        out_down.append(out_vae)
        
        #  Mid layers
        out_vae = self.mid_resnet1(out_vae, time)
        out_vae = self.mid_transformer(out_vae, out_encoder)
        out_vae = self.mid_resnet2(out_vae, time)
        
        #  Upsampling
        out_vae = self.up_resnet1(torch.cat([out_vae, out_down.pop()], dim=1), time)
        out_vae = self.up_resnet2(torch.cat([out_vae, out_down.pop()], dim=1), time)
        out_vae = self.up_resnet3(torch.cat([out_vae, out_down.pop()], dim=1), time)
        
        out_vae = self.up_in(out_vae)

        out_vae = self.up_block1(out_vae=out_vae, out_encoder=out_encoder, time=time, out_down=out_down)
        out_vae = self.up_block2(out_vae=out_vae, out_encoder=out_encoder, time=time, out_down=out_down)
        out_vae = self.up_block3(out_vae=out_vae, out_encoder=out_encoder, time=time, out_down=out_down)

        #  Output layer
        out_vae = self.out(out_vae)
        return out_vae

# print(Unet()(torch.randn(2, 4, 64, 64), torch.randn(2, 77, 768), torch.LongTensor([26])).shape)
