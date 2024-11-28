import torch
import torch.nn as nn
from FFESR.FFESR import FFESR, train, test
from FFESR.SCI1KDataset import SCI1KDataset
from FFESR.Args import args
import matplotlib.pyplot as plt

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
train(model, train_loader, optimizer, criterion, epochs=5, save_path=args.save_path)
compression_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
ssim_scores = []
for compression_ratio in compression_ratios:
    test_loader = torch.utils.data.DataLoader(SCI1KDataset(args.test_path, scale=compression_ratio), batch_size=1, shuffle=True)
    ssim_score = test(model, test_loader, criterion)
    ssim_scores.append(ssim_score)
# Plot the graph
plt.plot(compression_ratios, ssim_scores, marker='o')
plt.xlabel("Compression Rate")
plt.ylabel("Structural Similarity Index (SSIM)")
plt.title("SSIM vs. Compression Rate")
ax = plt.gca()
ax.set_ylim([min(ssim_scores), 1])
plt.grid(True)
plt.savefig(args.plot_path)
del model, optimizer, criterion, train_loader