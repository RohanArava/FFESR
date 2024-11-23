import os
import torch
from .Utils import get_hr_lr_images

class SCI1KDataset(torch.utils.data.Dataset):
    def __init__(self, path, scale=0.5):
      self.path = path
      self.paths = [os.path.join(path, name) for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
      self.scale = scale
    def __len__(self):
      return len(self.paths)
    def __getitem__(self, idx):
      hr_img, lr_img = get_hr_lr_images(self.paths[idx], self.scale)
      return hr_img, lr_img