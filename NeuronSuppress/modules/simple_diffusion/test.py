import torch


class Resnet(torch.nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.s = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=32,
                               num_channels=dim_in,
                               eps=1e-6,
                               affine=True),
            torch.nn.SiLU(),
            torch.nn.Conv2d(dim_in,
                            dim_out,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.GroupNorm(num_groups=32,
                               num_channels=dim_out,
                               eps=1e-6,
                               affine=True),
            torch.nn.SiLU(),
            torch.nn.Conv2d(dim_out,
                            dim_out,
                            kernel_size=3,
                            stride=1,
                            padding=1),
        )

        self.res = None
        if dim_in != dim_out:
            self.res = torch.nn.Conv2d(dim_in,
                                       dim_out,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

    def forward(self, x):
        #x -> [1, 128, 10, 10]

        res = x
        if self.res:
            #[1, 128, 10, 10] -> [1, 256, 10, 10]
            res = self.res(x)

        #[1, 128, 10, 10] -> [1, 256, 10, 10]
        return res + self.s(x)


# print(Resnet(128, 256)(torch.randn(1, 128, 10, 10)).shape)

class Atten(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.GroupNorm(num_channels=512,
                                       num_groups=32,
                                       eps=1e-6,
                                       affine=True)

        self.q = torch.nn.Linear(512, 512)
        self.k = torch.nn.Linear(512, 512)
        self.v = torch.nn.Linear(512, 512)
        self.out = torch.nn.Linear(512, 512)

    def forward(self, x):
        #x -> [1, 512, 64, 64]
        res = x

        #norm,维度不变
        #[1, 512, 64, 64]
        x = self.norm(x)

        #[1, 512, 64, 64] -> [1, 512, 4096] -> [1, 4096, 512]
        x = x.flatten(start_dim=2).transpose(1, 2)

        #线性运算,维度不变
        #[1, 4096, 512]
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        #[1, 4096, 512] -> [1, 512, 4096]
        k = k.transpose(1, 2)

        #[1, 4096, 512] * [1, 512, 4096] -> [1, 4096, 4096]
        #0.044194173824159216 = 1 / 512**0.5
        #atten = q.bmm(k) * 0.044194173824159216

        #照理来说应该是等价的,但是却有很小的误差
        atten = torch.baddbmm(torch.empty(1, 4096, 4096, device=q.device),
                              q,
                              k,
                              beta=0,
                              alpha=0.044194173824159216)

        atten = torch.softmax(atten, dim=2)

        #[1, 4096, 4096] * [1, 4096, 512] -> [1, 4096, 512]
        atten = atten.bmm(v)

        #线性运算,维度不变
        #[1, 4096, 512]
        atten = self.out(atten)

        #[1, 4096, 512] -> [1, 512, 4096] -> [1, 512, 64, 64]
        atten = atten.transpose(1, 2).reshape(-1, 512, 64, 64)

        #残差连接,维度不变
        #[1, 512, 64, 64]
        atten = atten + res

        return atten


# print(Atten()(torch.randn(1, 512, 64, 64)).shape)

class Pad(torch.nn.Module):

    def forward(self, x):
        return torch.nn.functional.pad(x, (0, 1, 0, 1),
                                       mode='constant',
                                       value=0)


# print(Pad()(torch.ones(1, 2, 5, 5)))

class VAE(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            #in
            torch.nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),

            #down
            torch.nn.Sequential(
                Resnet(128, 128),
                Resnet(128, 128),
                torch.nn.Sequential(
                    Pad(),
                    torch.nn.Conv2d(128, 128, 3, stride=2, padding=0),
                ),
            ),
            torch.nn.Sequential(
                Resnet(128, 256),
                Resnet(256, 256),
                torch.nn.Sequential(
                    Pad(),
                    torch.nn.Conv2d(256, 256, 3, stride=2, padding=0),
                ),
            ),
            torch.nn.Sequential(
                Resnet(256, 512),
                Resnet(512, 512),
                torch.nn.Sequential(
                    Pad(),
                    torch.nn.Conv2d(512, 512, 3, stride=2, padding=0),
                ),
            ),
            torch.nn.Sequential(
                Resnet(512, 512),
                Resnet(512, 512),
            ),

            #mid
            torch.nn.Sequential(
                Resnet(512, 512),
                Atten(),
                Resnet(512, 512),
            ),

            #out
            torch.nn.Sequential(
                torch.nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6),
                torch.nn.SiLU(),
                torch.nn.Conv2d(512, 8, 3, padding=1),
            ),

            #正态分布层
            torch.nn.Conv2d(8, 8, 1),
        )

        self.decoder = torch.nn.Sequential(
            #正态分布层
            torch.nn.Conv2d(4, 4, 1),

            #in
            torch.nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1),

            #middle
            torch.nn.Sequential(Resnet(512, 512), Atten(), Resnet(512, 512)),

            #up
            torch.nn.Sequential(
                Resnet(512, 512),
                Resnet(512, 512),
                Resnet(512, 512),
                torch.nn.Upsample(scale_factor=2.0, mode='nearest'),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ),
            torch.nn.Sequential(
                Resnet(512, 512),
                Resnet(512, 512),
                Resnet(512, 512),
                torch.nn.Upsample(scale_factor=2.0, mode='nearest'),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ),
            torch.nn.Sequential(
                Resnet(512, 256),
                Resnet(256, 256),
                Resnet(256, 256),
                torch.nn.Upsample(scale_factor=2.0, mode='nearest'),
                torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ),
            torch.nn.Sequential(
                Resnet(256, 128),
                Resnet(128, 128),
                Resnet(128, 128),
            ),

            #out
            torch.nn.Sequential(
                torch.nn.GroupNorm(num_channels=128, num_groups=32, eps=1e-6),
                torch.nn.SiLU(),
                torch.nn.Conv2d(128, 3, 3, padding=1),
            ),
        )

    def sample(self, h):
        #h -> [1, 8, 64, 64]

        #[1, 4, 64, 64]
        mean = h[:, :4]
        logvar = h[:, 4:]
        std = logvar.exp()**0.5

        #[1, 4, 64, 64]
        h = torch.randn(mean.shape, device=mean.device)
        h = mean + std * h

        return h

    def forward(self, x):
        #x -> [1, 3, 512, 512]

        #[1, 3, 512, 512] -> [1, 8, 64, 64]
        h = self.encoder(x)

        #[1, 8, 64, 64] -> [1, 4, 64, 64]
        h = self.sample(h)

        #[1, 4, 64, 64] -> [1, 3, 512, 512]
        h = self.decoder(h)

        return h


# print(VAE()(torch.randn(1, 3, 512, 512)).shape)


from diffusers import AutoencoderKL

#加载预训练模型的参数
pretrain_model_path = '/home/junlei/.cache/huggingface/lansinuote/diffsion_from_scratch.params'
params = AutoencoderKL.from_pretrained(pretrain_model_path, subfolder='vae', low_cpu_mem_usage=False, ignore_mismatched_sizes=True)


vae = VAE()


def load_res(model, param):
    model.s[0].load_state_dict(param.norm1.state_dict())
    model.s[2].load_state_dict(param.conv1.state_dict())
    model.s[3].load_state_dict(param.norm2.state_dict())
    model.s[5].load_state_dict(param.conv2.state_dict())

    if isinstance(model.res, torch.nn.Module):
        model.res.load_state_dict(param.conv_shortcut.state_dict())


def load_atten(model, param):
    model.norm.load_state_dict(param.group_norm.state_dict())
    model.q.load_state_dict(param.query.state_dict())
    model.k.load_state_dict(param.key.state_dict())
    model.v.load_state_dict(param.value.state_dict())
    model.out.load_state_dict(param.proj_attn.state_dict())


#encoder.in
vae.encoder[0].load_state_dict(params.encoder.conv_in.state_dict())

#encoder.down
for i in range(4):
    load_res(vae.encoder[i + 1][0], params.encoder.down_blocks[i].resnets[0])
    load_res(vae.encoder[i + 1][1], params.encoder.down_blocks[i].resnets[1])

    if i != 3:
        vae.encoder[i + 1][2][1].load_state_dict(
            params.encoder.down_blocks[i].downsamplers[0].conv.state_dict())

#encoder.mid
load_res(vae.encoder[5][0], params.encoder.mid_block.resnets[0])
load_res(vae.encoder[5][2], params.encoder.mid_block.resnets[1])
load_atten(vae.encoder[5][1], params.encoder.mid_block.attentions[0])

#encoder.out
vae.encoder[6][0].load_state_dict(params.encoder.conv_norm_out.state_dict())
vae.encoder[6][2].load_state_dict(params.encoder.conv_out.state_dict())

#encoder.正态分布层
vae.encoder[7].load_state_dict(params.quant_conv.state_dict())

#decoder.正态分布层
vae.decoder[0].load_state_dict(params.post_quant_conv.state_dict())

#decoder.in
vae.decoder[1].load_state_dict(params.decoder.conv_in.state_dict())

#decoder.mid
load_res(vae.decoder[2][0], params.decoder.mid_block.resnets[0])
load_res(vae.decoder[2][2], params.decoder.mid_block.resnets[1])
load_atten(vae.decoder[2][1], params.decoder.mid_block.attentions[0])

#decoder.up
for i in range(4):
    load_res(vae.decoder[i + 3][0], params.decoder.up_blocks[i].resnets[0])
    load_res(vae.decoder[i + 3][1], params.decoder.up_blocks[i].resnets[1])
    load_res(vae.decoder[i + 3][2], params.decoder.up_blocks[i].resnets[2])

    if i != 3:
        vae.decoder[i + 3][4].load_state_dict(
            params.decoder.up_blocks[i].upsamplers[0].conv.state_dict())

#decoder.out
vae.decoder[7][0].load_state_dict(params.decoder.conv_norm_out.state_dict())
vae.decoder[7][2].load_state_dict(params.decoder.conv_out.state_dict())

data = torch.randn(1, 3, 512, 512)

a = vae.encoder(data)
b = params.encode(data).latent_dist.parameters

print((a == b).all())