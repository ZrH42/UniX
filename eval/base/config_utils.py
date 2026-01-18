import json
import os


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file.

    Args:
        config_path: Path to config.json

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)