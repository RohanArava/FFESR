import os
import gc
import torch
import torch.nn as nn
from .EFE import EnhancedFeatureExtraction
from ..ITSRN.code import models
from .Args import args

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
    
def train(model, train_loader, optimizer, criterion, save_path=".", epochs=10):
  for epoch in range(epochs):
    for i, (hr_img, lr_img) in enumerate(train_loader):
      hr_img = hr_img.cuda()
      lr_img = lr_img.cuda()
      optimizer.zero_grad()
      
      hr_out, lr_out = model(lr_img, hr_img.shape[2:])
      loss1 = criterion(hr_out, hr_img)
      loss2 = criterion(lr_out, lr_img)
      loss = loss1 + loss2
      loss.backward()
      optimizer.step()
      print(f"Epoch: {epoch}, item: {i}, Loss: {loss.item()}")
      del hr_img, lr_img, hr_out, lr_out, loss1, loss2, loss
      gc.collect()
    torch.save(model, os.path.join(save_path, f"model_{epoch}.pth"))