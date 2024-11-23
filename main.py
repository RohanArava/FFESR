import torch
import torch.nn as nn
import gc
from FFESR.FFESR import FFESR, train
from FFESR.SCI1KDataset import SCI1KDataset

model = FFESR(
        input_channels=3,
        base_channels=3,
        num_rdb=3,
        num_dense_layers=3,
        growth_rate=16
    ).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.L1Loss()
train_loader = torch.utils.data.DataLoader(SCI1KDataset("/content/drive/MyDrive/HR"), batch_size=1, shuffle=True)
train(model, train_loader, optimizer, criterion, epochs=10)
del model, optimizer, criterion, train_loader
gc.collect()