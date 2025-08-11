import os
from PIL import Image
from torch.utils.data import Dataset


class Dataset(Dataset):

    def __init__(self, path, transform, H=256, W=256):
        super(Dataset, self).__init__()
        self.H = H
        self.W = W
        self.path = path
        self.list = os.listdir(path)
        self.transform = transform

    def transform_image(self, image):
        image = self.transform(image)

        return image

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.path, self.list[index])).convert("RGB")
        image = self.transform_image(image)
        
        return image

    def __len__(self):
        return len(self.list)
