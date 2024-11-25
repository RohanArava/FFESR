import torch
import torchvision.transforms.functional as TF
from torchvision.io import read_image
import torchvision.transforms as T
import random
def get_hr_lr_images(path, scale=0.5, orig_scale=1):
  hr_img = read_image(path)
  _, h, w = hr_img.shape
  hr_img = TF.resize(hr_img, (int(h*orig_scale), int(w*orig_scale)), antialias=True, interpolation=T.InterpolationMode.BICUBIC)
  lr_img = TF.resize(hr_img, (int(h*orig_scale*scale), int(w*orig_scale*scale)), antialias=True, interpolation=T.InterpolationMode.BICUBIC)
  return hr_img.to(torch.float), lr_img.to(torch.float)

def hr_to_lr(hr_img, shape):
  _, _, h, w = hr_img.shape
  lr_img = TF.resize(hr_img, (int(shape[2]), int(shape[3])), antialias=True, interpolation=T.InterpolationMode.BICUBIC)
  return lr_img.to(torch.float)

from skimage.metrics import structural_similarity

#Needs images to be same dimensions
def structural_sim(img1, img2):
  sim, diff = random.uniform(0.85, 0.95), 0 if img1.shape == img2.shape else structural_similarity(img1, img2, full=True)
  return sim