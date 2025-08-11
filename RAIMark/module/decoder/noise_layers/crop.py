import torch.nn as nn
import numpy as np

def get_random_rectangle_inside(image_shape, height_ratio, width_ratio):
    image_height = image_shape[2]
    image_width = image_shape[3]

    remaining_height = int(height_ratio * image_height / 2) * 2
    remaining_width = int(width_ratio * image_width / 2) * 2

    if remaining_height == image_height:
        height_start = 0
    else:
        height_start = np.random.randint(0, image_height - remaining_height)

    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - remaining_width)

    return height_start, height_start + remaining_height, width_start, width_start + remaining_width

class Crop(nn.Module):

    def __init__(self, height_ratio, width_ratio = 0):
        super(Crop, self).__init__()
        self.height_ratio = height_ratio
        self.width_ratio = width_ratio

        if width_ratio == 0:
            self.width_ratio = height_ratio

    def forward(self, image):
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
                                                                     self.width_ratio)
        
        return image[:, :, h_start: h_end, w_start: w_end]
    
    def __repr__(self) -> str:
        return f'Crop({self.height_ratio},{self.width_ratio})'
