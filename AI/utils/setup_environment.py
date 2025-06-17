import torch
import random
import numpy as np
from pathlib import Path


def set_seed(seed: int = 0):
    """
    Sets the random seed for reproducibility across different libraries.

    Args:
        seed (int): The seed value to use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global random seed set to {seed} for reproducibility.")


def create_output_directories(output_dir: Path):
    """
    Creates the necessary output directory for saving checkpoints and logs.

    Args:
        output_dir (Path): The Path object representing the desired output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output checkpoint directory created at: {output_dir}") 