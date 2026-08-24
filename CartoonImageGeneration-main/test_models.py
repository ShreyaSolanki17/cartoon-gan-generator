"""Smoke test for the model architectures. Plain asserts, no test framework.

Run: python test_models.py

Checks that Generator/Discriminator/AttributePredictor build, produce
correctly-shaped outputs on dummy data, and that a full forward+backward+
optimizer step produces finite (non-NaN/Inf) gradients and loss.
"""
import torch
import torch.nn as nn

from model import ATTR_MAXES, CartoonGenerator, AttributePredictor

BATCH_SIZE = 2  # >1 so BatchNorm layers work in train mode


# ponytail: copied from train.py rather than imported, because train.py pulls in
# torch.utils.tensorboard at module level, which this smoke test has no need for
# (same reason generate.py originally kept its own copy of the shared model code).
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1)
        )

    def forward(self, x):
        return self.main(x)


def random_attrs(batch_size):
    return torch.stack([
        torch.randint(0, max_val + 1, (batch_size,)) for max_val in ATTR_MAXES
    ], dim=1)


def all_finite(*tensors):
    return all(torch.isfinite(t).all() for t in tensors)


def test_shapes():
    gen = CartoonGenerator()
    disc = Discriminator()
    attr_pred = AttributePredictor()

    attrs = random_attrs(BATCH_SIZE)
    fake_img, fake_outline = gen(attrs)
    assert fake_img.shape == (BATCH_SIZE, 3, 128, 128), fake_img.shape
    assert fake_outline.shape == (BATCH_SIZE, 2, 128, 128), fake_outline.shape

    disc_out = disc(fake_img)
    assert disc_out.shape == (BATCH_SIZE, 1), disc_out.shape

    attr_preds = attr_pred(fake_img)
    assert len(attr_preds) == len(ATTR_MAXES)
    for pred, max_val in zip(attr_preds, ATTR_MAXES):
        assert pred.shape == (BATCH_SIZE, max_val + 1), (pred.shape, max_val)

    print("test_shapes passed")


def test_forward_backward_step():
    gen = CartoonGenerator()
    disc = Discriminator()
    attr_pred = AttributePredictor()
    opt_g = torch.optim.Adam(gen.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-3)
    opt_attr = torch.optim.Adam(attr_pred.parameters(), lr=1e-3)

    attrs = random_attrs(BATCH_SIZE)
    real_img = torch.randn(BATCH_SIZE, 3, 128, 128)
    real_outline = torch.randn(BATCH_SIZE, 2, 128, 128)

    fake_img, fake_outline = gen(attrs)

    d_loss = nn.BCEWithLogitsLoss()(disc(fake_img.detach()), torch.zeros(BATCH_SIZE, 1))
    opt_d.zero_grad()
    d_loss.backward()
    opt_d.step()
    assert all_finite(d_loss)

    attr_preds = attr_pred(fake_img.detach())
    attr_loss = sum(nn.CrossEntropyLoss()(pred, attrs[:, i]) for i, pred in enumerate(attr_preds))
    opt_attr.zero_grad()
    attr_loss.backward()
    opt_attr.step()
    assert all_finite(attr_loss)

    g_loss = nn.MSELoss()(fake_img, real_img) + nn.MSELoss()(fake_outline, real_outline)
    opt_g.zero_grad()
    g_loss.backward()
    opt_g.step()
    assert all_finite(g_loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in gen.parameters())

    print("test_forward_backward_step passed")


if __name__ == "__main__":
    test_shapes()
    test_forward_backward_step()
    print("All smoke tests passed.")
