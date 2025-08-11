import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from ..constant import *
from module import FuncImg


class ImgLoader():
    def __init__(self, imgname, loadwm=False, msg_length=30):
        img = Image.open("./{}/{}.jpg".format(SIMG, imgname)).convert("RGB")

        img = img.resize((256, 256))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])

        self.imgdata = self.transform(img).unsqueeze(0)

        self.C = len(img.getbands())

        if loadwm:
            data = torch.load(loadwm, map_location="cpu")

            self.watermark = data["watermark"]
            state_dict_wm = data["state_dict"]
        
        else:
            self.watermark = torch.Tensor(np.random.choice([-1, 1], (1, msg_length)))

        state_dict = torch.load("./{}/{}.pth".format(FSIMG, imgname), map_location="cpu")

        layer_ids = [int(x.split(".")[1]) for x in state_dict.keys()]

        hidden_layers = max(layer_ids) // 2 - 1

        self.funcimg = FuncImg(out_channels=self.C, hidden_layers=hidden_layers)
        self.funcimg.load_state_dict(state_dict)

        if loadwm:
            self.funcimg_wm = FuncImg(out_channels=self.C, hidden_layers=hidden_layers)
            self.funcimg_wm.load_state_dict(state_dict_wm)
