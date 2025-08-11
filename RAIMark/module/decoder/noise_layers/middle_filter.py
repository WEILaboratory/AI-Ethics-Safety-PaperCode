import torch.nn as nn
from kornia.filters import MedianBlur


class MF(nn.Module):

    def __init__(self, kernel):
        super(MF, self).__init__()
        self.kernel = kernel
        self.middle_filter = MedianBlur((kernel, kernel))

    def forward(self, image):
        return self.middle_filter(image)
    
    def __repr__(self) -> str:
        return f'MF({self.kernel})'

