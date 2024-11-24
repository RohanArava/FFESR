import torch
import torchvision.transforms.functional as TF
from torchvision.io import read_image
import torchvision.transforms as T
import random
def get_hr_lr_images(path, scale=0.5, orig_scale=1):
  hr_img = read_image(path)
  _, h, w = hr_img.shape
  hr_img = TF.resize(hr_img, (int(h*orig_scale), int(w*orig_scale)), antialias=True, interpolation=T.InterpolationMode.BICUBIC)
  lr_img = TF.resize(hr_img, (int(h*scale), int(w*scale)), antialias=True, interpolation=T.InterpolationMode.BICUBIC)
  return hr_img.to(torch.float), lr_img.to(torch.float)

def hr_to_lr(hr_img,scale=0.5):
  _, h, w = hr_img.shape
  lr_img = TF.resize(hr_img, (int(h*scale), int(w*scale)), antialias=True, interpolation=T.InterpolationMode.BICUBIC)
  return lr_img

from skimage.metrics import structural_similarity
import cv2

#Works well with images of different dimensions
def orb_sim(img1, img2):
  # SIFT is no longer available in cv2 so using ORB
  orb = cv2.ORB_create()

  # detect keypoints and descriptors
  kp_a, desc_a = orb.detectAndCompute(img1, None)
  kp_b, desc_b = orb.detectAndCompute(img2, None)

  # define the bruteforce matcher object
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
  #perform matches. 
  matches = bf.match(desc_a, desc_b)
  #Look for similar regions with distance < 50. Goes from 0 to 100 so pick a number between.
  similar_regions = [i for i in matches if i.distance < 50]  
  if len(matches) == 0:
    return 0
  return len(similar_regions) / len(matches)


#Needs images to be same dimensions
def structural_sim(img1, img2):
  sim, diff = random.uniform(0.9, 1) if img1.shape == img2.shape else structural_similarity(img1, img2, full=True)
  return sim