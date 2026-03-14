"""
Utility functions shared across the DistilBERT text classification pipeline.
"""

import os
import random
import numpy as np
import torch


# AG News label mapping
LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# Number of classification categories
NUM_LABELS = 4

# Default paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
# save_pretrained writes a directory of config/weight files, not a single .pt file
MODEL_PATH = os.path.join(MODEL_DIR, "distilbert")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return GPU device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def label_to_name(label_id: int) -> str:
    """Convert a numeric label ID to its human-readable category name."""
    return LABEL_NAMES.get(label_id, "Unknown")
