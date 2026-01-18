"""Evaluation data directory.

This directory contains evaluation datasets for the UniX model.

Structure:
    understanding/
        test_data.json - Image understanding evaluation data (VQA format)
    generation/
        generations_with_metadata.csv - Text-to-image generation prompts

Usage:
    from eval.data import get_understanding_data_path, get_generation_data_path

Default paths:
    understanding: eval/data/understanding/test_data.json
    generation: eval/data/generation/generations_with_metadata.csv
"""

import os

# Default data paths
DEFAULT_UNDERSTANDING_DATA = os.path.join(
    os.path.dirname(__file__),
    "understanding",
    "test_data.json"
)

DEFAULT_GENERATION_DATA = os.path.join(
    os.path.dirname(__file__),
    "generation",
    "generations_with_metadata.csv"
)


def get_understanding_data_path(path: str = None) -> str:
    """Get path to understanding evaluation data.

    Args:
        path: Custom path, or None for default

    Returns:
        Path to data file
    """
    if path is None:
        return DEFAULT_UNDERSTANDING_DATA
    return path


def get_generation_data_path(path: str = None) -> str:
    """Get path to generation evaluation data.

    Args:
        path: Custom path, or None for default

    Returns:
        Path to data file
    """
    if path is None:
        return DEFAULT_GENERATION_DATA
    return path


__all__ = [
    "get_understanding_data_path",
    "get_generation_data_path",
    "DEFAULT_UNDERSTANDING_DATA",
    "DEFAULT_GENERATION_DATA",
]
