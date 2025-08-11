import torch.nn as nn
import torch.nn.functional as F


class Resize(nn.Module):
    """
    Resize the image.
    """
    def __init__(self, scale):
        super(Resize, self).__init__()

        self.side_scale = scale

    def forward(self, image):
        x = F.interpolate(image, scale_factor=(self.side_scale, self.side_scale), mode="nearest")
        
        return x

    def __repr__(self):
        return "Resize({},{})".format(self.side_scale, self.side_scale)
