import torch
import torch.nn as nn
from torchvision import transforms

import augly.image.functional as aug_functional


class Jpeg(nn.Module):
    def __init__(self, Q):
        super(Jpeg, self).__init__()
        self.Q = Q

        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def forward(self, image):
        with torch.no_grad():
            img_clip = image.clamp(-1, 1)

            pil_img = self.to_pil(img_clip[0])
            img_aug = self.to_tensor(aug_functional.encoding_quality(pil_img, quality=self.Q))
            img_aug = img_aug.unsqueeze(0)

            img_aug = img_aug.to(image.device)

            img_gap = img_aug - image
            img_gap = img_gap.detach()

        img_aug = image + img_gap

        return img_aug
    
    def __repr__(self):
        return "Jpeg(Q=" + str(self.Q) + ")"
