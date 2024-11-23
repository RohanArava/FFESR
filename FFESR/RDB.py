import torch
import torch.nn as nn
def make_dense(nChannels, growthRate):
    return nn.Sequential(
        nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False),
        nn.ReLU(inplace=True)
    )

class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.ModuleList(modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        features = [x]
        for layer in self.dense_layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        out = self.conv_1x1(torch.cat(features, 1))
        return out + x