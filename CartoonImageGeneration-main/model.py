"""Shared model definitions used by both train.py and generate.py.

Kept dependency-free (no tensorboard/torchvision.transforms.functional/etc.)
so generate.py can import it without pulling in training-only packages.
"""
import torch
import torch.nn as nn

ATTR_MAXES = [3, 2, 2, 3, 2, 14, 4, 7, 15, 111, 5, 11, 10, 12, 7, 3, 3, 3]
FIXED_ATTRS = [0, 1, 0, 0, 1, 8, 2, 6, 0, 94, 1, 10, 6, 8, 1, 1, 1, 2]


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1), nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1), nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return torch.relu(x + self.conv(x))


class CartoonGenerator(nn.Module):
    def __init__(self):
        super(CartoonGenerator, self).__init__()
        self.attr_encoder = nn.Sequential(
            nn.Linear(18, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        self.assembler = nn.Sequential(
            nn.Linear(128, 128 * 8 * 8), nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8))
        )
        self.structure = nn.Sequential(
            nn.ConvTranspose2d(128 + 64, 96, 4, 2, 1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.ConvTranspose2d(96 + 32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64 + 16, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32 + 8, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 2, 3, 1, 1), nn.Sigmoid()
        )
        self.texture = nn.Sequential(
            nn.ConvTranspose2d(128 + 64, 96, 4, 2, 1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.ConvTranspose2d(96 + 32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64 + 16, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32 + 8, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            ResidualBlock(16),
            ResidualBlock(16),
            nn.Conv2d(16, 3, 3, 1, 1)
        )
        self.attr_proj1 = nn.Linear(18, 64 * 8 * 8)
        self.attr_proj2 = nn.Linear(18, 32 * 16 * 16)
        self.attr_proj3 = nn.Linear(18, 16 * 32 * 32)
        self.attr_proj4 = nn.Linear(18, 8 * 64 * 64)
        self.hair_beard_proj = nn.Linear(2, 16 * 128 * 128)
        self.fusion_down = nn.Sequential(
            nn.Conv2d(5, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU()
        )
        self.fusion_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1)
        )

    def forward(self, attrs):
        attrs_flat = attrs.float() / torch.tensor(ATTR_MAXES, device=attrs.device).float()
        attr_features = self.attr_encoder(attrs_flat)
        canvas = self.assembler(attr_features)

        attr_8x8 = self.attr_proj1(attrs_flat).view(-1, 64, 8, 8)
        attr_16x16 = self.attr_proj2(attrs_flat).view(-1, 32, 16, 16)
        attr_32x32 = self.attr_proj3(attrs_flat).view(-1, 16, 32, 32)
        attr_64x64 = self.attr_proj4(attrs_flat).view(-1, 8, 64, 64)

        x = torch.cat([canvas, attr_8x8], dim=1)
        x = self.structure[0:3](x)
        x = torch.cat([x, attr_16x16], dim=1)
        x = self.structure[3:6](x)
        x = torch.cat([x, attr_32x32], dim=1)
        x = self.structure[6:9](x)
        x = torch.cat([x, attr_64x64], dim=1)
        outline = self.structure[9:](x)

        x = torch.cat([canvas, attr_8x8], dim=1)
        x = self.texture[0:3](x)
        x = torch.cat([x, attr_16x16], dim=1)
        x = self.texture[3:6](x)
        x = torch.cat([x, attr_32x32], dim=1)
        x = self.texture[6:9](x)
        x = torch.cat([x, attr_64x64], dim=1)
        x = self.texture[9:12](x)
        hair_beard_attrs = attrs_flat[:, [8, 9]].view(-1, 2)
        hair_beard_map = self.hair_beard_proj(hair_beard_attrs).view(-1, 16, 128, 128)
        x = x + 0.1 * hair_beard_map
        color = self.texture[12:](x)

        combined = torch.cat([outline, color], dim=1)
        x = self.fusion_down(combined)
        x = self.fusion_up(x)
        output = torch.tanh(x + color)

        return output, outline


class AttributePredictor(nn.Module):
    def __init__(self):
        super(AttributePredictor, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256), nn.ReLU()
        )
        self.heads = nn.ModuleList([nn.Linear(256, max_val + 1) for max_val in ATTR_MAXES])

    def forward(self, x):
        features = self.main(x)
        return [head(features) for head in self.heads]
