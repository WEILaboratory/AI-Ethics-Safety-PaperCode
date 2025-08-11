from torchvision import transforms
from PIL import Image

from ..constant import *


class ImgDataset():
    def __init__(self, H, W, imgname):
        img = Image.open("./{}/{}".format(SIMG, imgname)).convert("RGB")

        if H and W:
            img = img.resize((H, W))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])

        self.imgdata = self.transform(img).unsqueeze(0)
 
        self.C, self.H, self.W = self.imgdata.shape[1:]

    def __len__(self):
        return self.H * self.W
