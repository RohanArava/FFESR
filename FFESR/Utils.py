import torch
import torchvision.transforms.functional as TF
from torchvision.io import read_image
import torchvision.transforms as T
def get_hr_lr_images(path, scale=0.5, orig_scale=1):
  hr_img = read_image(path)
  _, h, w = hr_img.shape
  hr_img = TF.resize(hr_img, (int(h*orig_scale), int(w*orig_scale)), antialias=True, interpolation=T.InterpolationMode.BICUBIC)
  lr_img = TF.resize(hr_img, (int(h*scale), int(w*scale)), antialias=True, interpolation=T.InterpolationMode.BICUBIC)
  return hr_img.to(torch.float), lr_img.to(torch.float)

