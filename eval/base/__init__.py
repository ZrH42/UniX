"""Base utilities for evaluation.

Modules:
    model_utils: Common model initialization and configuration
    worker: Parallel worker processes for distributed evaluation
    dataset: Dataset classes for evaluation data loading
"""

from .model_utils import (
    load_config,
    get_model_paths_from_config,
    get_inference_config,
    initialize_model_on_gpu,
    DEFAULT_VAE_MAX_SIZE,
    DEFAULT_VAE_MIN_SIZE,
    DEFAULT_VAE_STRIDE,
    DEFAULT_VIT_MAX_SIZE,
    DEFAULT_VIT_MIN_SIZE,
    DEFAULT_VIT_STRIDE,
    DEFAULT_MAX_MEM_PER_GPU,
)

from .worker import (
    BaseWorker,
    UnderstandingWorker,
    GenerationWorker,
    setup_random_seed,
    run_workers,
)

from .dataset import (
    UnderstandingDataset,
    GenerationDataset,
    load_understanding_data,
    load_generation_data,
)

__all__ = [
    "load_config",
    "get_model_paths_from_config",
    "get_inference_config",
    "initialize_model_on_gpu",
    "BaseWorker",
    "UnderstandingWorker",
    "GenerationWorker",
    "setup_random_seed",
    "run_workers",
    "UnderstandingDataset",
    "GenerationDataset",
    "load_understanding_data",
    "load_generation_data",
    "DEFAULT_VAE_MAX_SIZE",
    "DEFAULT_VAE_MIN_SIZE",
    "DEFAULT_VAE_STRIDE",
    "DEFAULT_VIT_MAX_SIZE",
    "DEFAULT_VIT_MIN_SIZE",
    "DEFAULT_VIT_STRIDE",
    "DEFAULT_MAX_MEM_PER_GPU",
]
