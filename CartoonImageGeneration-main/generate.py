"""Inference script: generate a cartoon face from 18 facial attribute values
using a trained CartoonGenerator checkpoint.

Usage:
    python generate.py --checkpoint models/gen_final.pth --out samples/face
    python generate.py --checkpoint models/gen_final.pth --attrs 0 1 0 0 1 8 2 6 0 94 1 10 6 8 1 1 1 2
"""
import argparse
import os

import torch
from torchvision.transforms.functional import to_pil_image

from model import ATTR_MAXES, FIXED_ATTRS, CartoonGenerator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    color_img.save(f"{args.out}_color.png")
    face_outline.save(f"{args.out}_face_outline.png")
    hair_outline.save(f"{args.out}_hair_outline.png")
    print(f"Saved {args.out}_color.png, {args.out}_face_outline.png, {args.out}_hair_outline.png")
