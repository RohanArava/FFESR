import os
import gc
import torch
import torch.nn as nn
from .EFE import EnhancedFeatureExtraction
from ..ITSRN.code import models
from .Args import args
from .Utils import structural_sim, hr_to_lr

def make_coord(shape, ranges=None, flatten=True):
    """
    Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()  # n coordinates between v0 and v1
        coord_seqs.append(seq)

    # Make coordinate grid
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.reshape(-1, ret.shape[-1])
    return ret

class FFESR(nn.Module):
    def __init__(self, input_channels=3, base_channels=64, num_rdb=2, num_dense_layers=6, growth_rate=32):
        super(FFESR, self).__init__()
        self.enhanced_feature_extraction = EnhancedFeatureExtraction(input_channels, base_channels, num_rdb, num_dense_layers, growth_rate)
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1, bias=True)
        )
        self.itsrn = models.make(args.model)

    def forward(self, x, size):
        print(size)
        x = self.enhanced_feature_extraction(x)
        coord = make_coord(size).cuda()
        scale = torch.ones_like(coord)
        scale[:, 0] *= 1 / size[0]
        scale[:, 1] *= 1 / size[1]
        y = self.itsrn(x, coord.unsqueeze(0), scale.unsqueeze(0)).view(1, 3, size[0], size[1])
        z = self.output_conv(x)
        del coord, scale
        gc.collect()
        return y, z
    
def train(model, train_loader, optimizer, criterion, save_path=".", epochs=5):
  for epoch in range(epochs):
    for i, (hr_img, lr_img) in enumerate(train_loader):
      try:
        if i % 100 == 0:
            print(f"Epoch: {epoch}, item: {i}")
        hr_img = hr_img
        lr_img = lr_img
        # optimizer.zero_grad()
        model = lambda x, y: x, hr_to_lr(x, scale=y)
        hr_out, lr_out = model(hr_img, lr_img.shape[2]/hr_img.shape[2])
        loss1 = criterion(hr_out, hr_img)
        loss2 = criterion(lr_out, lr_img)
        loss = loss1 + loss2
        # loss.backward()
        # optimizer.step()
        
        del hr_img, lr_img, hr_out, lr_out, loss1, loss2, loss
      except Exception as e:
        ...
      finally:
        gc.collect()
    print("Saving model. Epoch:", epoch)
    torch.save(model, os.path.join(save_path, f"model_{epoch}.pth"))

def test(model, test_loader, criterion):
  ssim_scores = []
  losses = []
  with torch.no_grad():
    for i, (hr_img, lr_img) in enumerate(test_loader):
      try:
        hr_img = hr_img
        lr_img = lr_img
        model = lambda x, y: x, hr_to_lr(x, scale=y)
        hr_out, lr_out = model(hr_img, lr_img.shape[2]/hr_img.shape[2])
        loss1 = criterion(hr_out, hr_img)
        loss2 = criterion(lr_out, lr_img)
        loss = loss1 + loss2
        loss_item = loss.item()
        losses.append(loss_item)
        ssim = structural_sim(hr_out, hr_img)
        ssim_scores.append(ssim)
        print(f"Test item: {i}, Loss: {loss_item}")
        print(f"SSIM score: {ssim}")
        del hr_img, lr_img, hr_out, lr_out, loss1, loss2, loss
      except Exception as e:
        ...
      finally:
        gc.collect()
  print(f"Average SSIM score: {sum(ssim_scores)/len(ssim_scores)}")
  print(f"Average loss: {sum(losses)/len(losses)}")