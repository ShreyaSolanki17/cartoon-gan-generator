"""Inference script: generate a cartoon face from 18 facial attribute values
using a trained CartoonGenerator checkpoint.

# ponytail: CartoonGenerator/ResidualBlock/ATTR_MAXES/FIXED_ATTRS are duplicated
# from train.py rather than imported, because train.py imports
# torch.utils.tensorboard at module level (a training-only dependency this
# inference script has no need for). Task 5 (config/model extraction) is the
# place to move the shared model code into its own module and delete this copy.

Usage:
    python generate.py --checkpoint models/gen_final.pth --out samples/face
    python generate.py --checkpoint models/gen_final.pth --attrs 0 1 0 0 1 8 2 6 0 94 1 10 6 8 1 1 1 2
"""
import argparse

import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image

ATTR_MAXES = [3, 2, 2, 3, 2, 14, 4, 7, 15, 111, 5, 11, 10, 12, 7, 3, 3, 3]
FIXED_ATTRS = [0, 1, 0, 0, 1, 8, 2, 6, 0, 94, 1, 10, 6, 8, 1, 1, 1, 2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def load_generator(checkpoint_path, device=DEVICE):
    """Load a trained CartoonGenerator from a .pth state_dict checkpoint, in eval mode."""
    gen = CartoonGenerator().to(device)
    gen.load_state_dict(torch.load(checkpoint_path, map_location=device))
    gen.eval()
    return gen


def generate_image(gen, attrs, device=DEVICE):
    """Run one forward pass for a single set of 18 raw attribute indices
    (same scheme as the CartoonSet100k CSVs / ATTR_MAXES above).

    Returns (color_img, face_outline, hair_outline) as PIL Images.
    """
    attrs_t = torch.tensor(attrs, dtype=torch.int64, device=device).unsqueeze(0)
    with torch.no_grad():
        fake_img, fake_outline = gen(attrs_t)

    # Models were trained against Normalize((0.5,), (0.5,)) targets, so undo that
    # the same way train.py's own TensorBoard visualization does.
    color_img = to_pil_image((fake_img.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1))
    face_outline = to_pil_image((fake_outline[:, 0:1].squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1))
    hair_outline = to_pil_image((fake_outline[:, 1:2].squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1))
    return color_img, face_outline, hair_outline


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a cartoon face from 18 facial attributes.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained generator .pth file")
    parser.add_argument(
        "--attrs", type=int, nargs=18, metavar="ATTR",
        help="18 attribute indices, in the order/range used by ATTR_MAXES in train.py "
             f"(per-slot max: {ATTR_MAXES}). Defaults to train.py's FIXED_ATTRS sample.",
    )
    parser.add_argument("--out", default="output", help="Output file prefix (default: output)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    attrs = args.attrs if args.attrs is not None else FIXED_ATTRS

    for i, (val, max_val) in enumerate(zip(attrs, ATTR_MAXES)):
        if not 0 <= val <= max_val:
            raise ValueError(f"attrs[{i}]={val} out of range [0, {max_val}]")

    gen = load_generator(args.checkpoint)
    color_img, face_outline, hair_outline = generate_image(gen, attrs)

    color_img.save(f"{args.out}_color.png")
    face_outline.save(f"{args.out}_face_outline.png")
    hair_outline.save(f"{args.out}_hair_outline.png")
    print(f"Saved {args.out}_color.png, {args.out}_face_outline.png, {args.out}_hair_outline.png")
