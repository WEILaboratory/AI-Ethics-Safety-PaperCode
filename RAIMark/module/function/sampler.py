import torch
import torch.nn as nn


class Sampler(nn.Module):
    def __init__(self, H, W):
        super(Sampler, self).__init__()

        self.H = H
        self.W = W

        hdata = torch.tensor([i for i in range(H)], dtype=torch.float32).repeat_interleave(W) / H * 2 - 1
        wdata = torch.tensor([i for i in range(W)], dtype=torch.float32).repeat(H) / W * 2 - 1

        self.axisdata = torch.stack([hdata, wdata], dim=1)

    def forward(self, funcimg, device=None):
        output = funcimg(self.axisdata)

        output = output.transpose(1, 0).reshape(-1, self.H, self.W)

        output = output.unsqueeze(0)

        return output
    
    def to(self, device):
        self.axisdata = self.axisdata.to(device)
        return self
    
    def __repr__(self):
        return "Sampler({}, {})".format(self.H, self.W)
