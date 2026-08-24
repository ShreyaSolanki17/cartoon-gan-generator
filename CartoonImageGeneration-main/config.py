"""Central configuration: dataset paths, attribute encoding, and training
hyperparameters shared across preProcess.py, train.py, and model.py.

Values here are defaults — train.py's CLI flags (--data-dir, --output-dir,
--epochs) override the path/epoch ones per run.
"""
import torch

# Facial attribute encoding (CartoonSet100k CSV column order): per-slot max
# value, and one fixed attribute vector used as a training-time preview sample.
ATTR_MAXES = [3, 2, 2, 3, 2, 14, 4, 7, 15, 111, 5, 11, 10, 12, 7, 3, 3, 3]
FIXED_ATTRS = [0, 1, 0, 0, 1, 8, 2, 6, 0, 94, 1, 10, 6, 8, 1, 1, 1, 2]

# preProcess.py: raw dataset -> outlines + tensors
RAW_IMAGE_DIR = "cartoonset100k/cartoonset100k"
OUTLINE_DIR = "cartoonset100k_outlines"
TENSOR_DIR = "cartoonset100k_tensors"
NUM_FOLDERS = 10

# train.py: output locations and hyperparameters
MODEL_SAVE_DIR = "models"
LOG_DIR = "runs"
BATCH_SIZE = 32
NUM_EPOCHS = 200
LEARNING_RATE_G = 5e-4
LEARNING_RATE_D = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator loss weight schedule: (epoch_upper_bound, weights), first matching
# bound wins (None = catch-all). weights = (outline, attr, color, perceptual, adv, hair_texture)
LOSS_WEIGHT_SCHEDULE = [
    (50, (0.35, 0.25, 0.15, 0.15, 0.10, 0.05)),
    (100, (0.30, 0.25, 0.15, 0.15, 0.10, 0.05)),
    (None, (0.25, 0.20, 0.20, 0.20, 0.10, 0.05)),
]


def loss_weights_for_epoch(epoch):
    for upper, weights in LOSS_WEIGHT_SCHEDULE:
        if upper is None or epoch < upper:
            return weights
