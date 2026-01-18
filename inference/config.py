from dataclasses import dataclass
import json
import os

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class ModelConfig:
    """Inference model configuration - All components bundled in UniX directory"""
    model_path: str = os.path.join(PROJECT_ROOT, "weights", "UniX")
    vae_path: str = None  # Defaults to model_path/vae/kl-f16d16.ckpt in pipeline.py
    config_path: str = None

    @classmethod
    def from_json(cls, config_path: str = None) -> "ModelConfig":
        """Load model configuration from JSON file.

        Args:
            config_path: Path to config.json (defaults to inference/config.json)

        Returns:
            ModelConfig instance
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.json")

        with open(config_path, 'r') as f:
            config = json.load(f)

        return cls(
            model_path=config.get('model_path', os.path.join(PROJECT_ROOT, "weights", "UniX")),
            vae_path=config.get('vae_path'),
            config_path=config_path,
        )
