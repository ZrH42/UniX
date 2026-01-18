import random

import torch


def setup_random_seed(seed: int):
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)


def denormalize_vae_output(image: torch.Tensor, use_gen_normalization: bool = True) -> torch.Tensor:
    """Denormalize VAE output from [-1, 1] to [0, 1] range.

    Args:
        image: Input tensor in normalized range
        use_gen_normalization: Whether generation normalization was used

    Returns:
        Denormalized image tensor in [0, 1] range
    """
    if use_gen_normalization:
        image = image / 2 + 0.5
    return image.clamp(0, 1)
