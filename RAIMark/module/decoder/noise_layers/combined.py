import torch.nn as nn


class Combined(nn.Module):
    def __init__(self, noises) -> None:
        super(Combined, self).__init__()

        self.noises = noises

    def forward(self, image):
        out = image
        for noise in self.noises:
            out = noise(out)

        return out
    
    def __repr__(self) -> str:
        return f"Combined({self.noises})"
