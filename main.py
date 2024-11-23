import torch
import torch.nn as nn
import gc
from FFESR.FFESR import FFESR, train
from FFESR.SCI1KDataset import SCI1KDataset
from FFESR.Args import args

model = FFESR(
        input_channels=3,
        base_channels=3,
        num_rdb=3,
        num_dense_layers=3,
        growth_rate=16
    ).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.L1Loss()
train_loader = torch.utils.data.DataLoader(SCI1KDataset(args.path), batch_size=1, shuffle=True)
train(model, train_loader, optimizer, criterion, epochs=10)
del model, optimizer, criterion, train_loader
gc.collect()

# hr_img, lr_img = get_hr_lr_images("data\HR\Dino.jpg", orig_scale=0.5)
# sample_input = lr_img.to(torch.float).unsqueeze(0).cuda()
# print(hr_img.shape)
# # Forward pass
# hr_out, lr_out = model(sample_input, hr_img.shape[1:])
# print(f"HR shape: {hr_img.size}")
# print(f"Input shape: {sample_input.shape}")
# print(f"Output shape: {hr_out.shape}, {lr_out.shape}")