import torch
import torch.nn as nn
from .RDB import RDB

class EnhancedFeatureExtraction(nn.Module):
    def __init__(self, input_channels=3, base_channels=64, num_rdb=2, num_dense_layers=6, growth_rate=32):
        super(EnhancedFeatureExtraction, self).__init__()

        # Shallow feature extraction
        self.sfe1 = nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1, bias=False)
        self.sfe2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False)

        # RDB blocks for local feature extraction
        self.rdbs = nn.ModuleList([
            RDB(base_channels, num_dense_layers, growth_rate)
            for _ in range(num_rdb)
        ])

        # Enhanced Feature Fusion
        total_rdb_channels = base_channels * (num_rdb + 1)  # +1 for initial features

        # First fusion layer - reduces to half of total dimensions
        self.fusion1 = nn.Sequential(
            nn.Conv2d(total_rdb_channels, total_rdb_channels // 2, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )

        # Fusion-enhanced layer - further reduces dimension to base_channels
        self.fusion2 = nn.Sequential(
            nn.Conv2d(total_rdb_channels // 2, base_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )

        # Context Enhancement Block
        self.context_enhancement = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.ReLU(inplace=True)
        )

        # Final conv
        self.conv_out = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # Shallow feature extraction
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        # Local feature extraction with RDBs
        rdb_in = sfe2
        local_features = [sfe2]  # Include shallow features

        for rdb in self.rdbs:
            rdb_out = rdb(rdb_in)
            local_features.append(rdb_out)
            rdb_in = rdb_out

        # Enhanced Feature Fusion
        # 1. Concatenate all local features
        concat_features = torch.cat(local_features, 1)
        # 2. First fusion - reduce to half dimension
        fused1 = self.fusion1(concat_features)
        # 3. Enhanced fusion - further reduce dimension
        fused2 = self.fusion2(fused1)

        # Context enhancement
        enhanced = self.context_enhancement(fused2)

        # Final output
        out = self.conv_out(enhanced)

        return out