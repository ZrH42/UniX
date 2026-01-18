"""Common model utilities for evaluation.

This module provides shared model initialization and configuration utilities
used across different evaluation scripts.

Functions:
    load_config: Load configuration from JSON file
    get_model_paths_from_config: Extract model paths from config.json
    get_inference_config: Get inference hyperparameters from config
    initialize_model_on_gpu: Initialize model and components on a specific GPU
"""

import os
import json
from typing import Tuple, Any, Dict, Optional

import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch

from modeling.unix_vlm_manager import UnixVLMComponentManager
from modeling.unix import UniXConfig, UniX, Qwen2ForCausalLM
from modeling.autoencoder import load_ae
from data.transforms import SigLIPImageTransform, VAEImageTransform


# Default transform parameters
DEFAULT_VAE_MAX_SIZE = 256
DEFAULT_VAE_MIN_SIZE = 256
DEFAULT_VAE_STRIDE = 16
DEFAULT_VIT_MAX_SIZE = 384
DEFAULT_VIT_MIN_SIZE = 384
DEFAULT_VIT_STRIDE = 16
DEFAULT_MAX_MEM_PER_GPU = "40GiB"


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file.

    Args:
        config_path: Path to config.json

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def get_model_paths_from_config(config_path: str) -> Dict[str, str]:
    """Extract model paths from config.json.

    Args:
        config_path: Path to config.json

    Returns:
        Dictionary with model_path, vae_path
    """
    config = load_config(config_path)
    return {
        'model_path': config.get('model_path', '/path/to/your/weights/UniX'),
        'vae_path': config.get('vae_path'),
    }


def get_inference_config(config_path: str, task: str = 'understanding') -> Dict:
    """Get inference hyperparameters from config.

    Args:
        config_path: Path to config.json
        task: Task type ('understanding' or 'generation')

    Returns:
        Inference configuration dictionary
    """
    full_config = load_config(config_path)
    return full_config.get("inference_config", {}).get(task, {}).copy()


def initialize_model_on_gpu(
    model_path: str,
    gpu_id: int,
    max_mem_per_gpu: str = DEFAULT_MAX_MEM_PER_GPU,
    vae_max_size: int = DEFAULT_VAE_MAX_SIZE,
    vae_min_size: int = DEFAULT_VAE_MIN_SIZE,
    vae_stride: int = DEFAULT_VAE_STRIDE,
    vit_max_size: int = DEFAULT_VIT_MAX_SIZE,
    vit_min_size: int = DEFAULT_VIT_MIN_SIZE,
    vit_stride: int = DEFAULT_VIT_STRIDE,
    use_vae_transform: bool = True,
    config_path: str = None,
    vae_path: str = None,
) -> Tuple[Any, Any, Any, Dict, Any, Any]:
    """Initialize model and components on a specific GPU.

    This function loads the UniX model, VAE, tokenizer, and transforms
    required for inference on a single GPU.

    Args:
        model_path: Path to UniX directory containing all components
        gpu_id: GPU device ID to use
        max_mem_per_gpu: Maximum memory to allocate per GPU
        vae_max_size: VAE transform max size
        vae_min_size: VAE transform min size
        vae_stride: VAE transform stride
        vit_max_size: ViT transform max size
        vit_min_size: ViT transform min size
        vit_stride: ViT transform stride
        use_vae_transform: Whether to use VAEImageTransform (True) or SigLIPImageTransform (False) for VAE
        config_path: Path to config.json (defaults to inference/config.json)
        vae_path: Path to VAE checkpoint (defaults to model_path/vae/)

    Returns:
        Tuple of (model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform)
    """
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    # All components are in model_path directory
    llm_path = model_path
    vit_path = model_path

    # Default VAE path if not provided
    if vae_path is None:
        vae_path = os.path.join(model_path, "vae", "kl-f16d16.ckpt")

    unix_manager = UnixVLMComponentManager(llm_path, vit_path)
    llm_config = unix_manager.create_language_model_config()
    vit_model = unix_manager.load_vision_model()

    # Only load VAE for generation tasks (when use_vae_transform=True)
    if use_vae_transform:
        vae_model, vae_config = load_ae(local_path=vae_path)
        vae_model = vae_model.to(dtype=torch.bfloat16)
    else:
        # For understanding tasks, VAE is not needed
        vae_model, vae_config = None, None

    language_model = Qwen2ForCausalLM(llm_config)

    # Create UniX config
    vit_config = unix_manager.create_dummy_vit_config()
    unix_config = UniXConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=24,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=1,
        max_latent_size=32,
        interpolate_pos=False,
        timestep_shift=1.0,
    )

    model = UniX(language_model, vit_model, unix_config)

    tokenizer = unix_manager.load_tokenizer()
    new_token_ids = unix_manager.get_new_token_ids()
    unix_manager.verify_token_ids()

    model = model.to(dtype=torch.bfloat16)
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if module.weight.dtype != torch.bfloat16:
                module.weight.data = module.weight.data.to(dtype=torch.bfloat16)
        if hasattr(module, 'bias') and module.bias is not None:
            if module.bias.dtype != torch.bfloat16:
                module.bias.data = module.bias.data.to(dtype=torch.bfloat16)

    device_map = infer_auto_device_map(
        model,
        max_memory={gpu_id: max_mem_per_gpu},
        no_split_module_classes=["UniX", "Qwen2MoTDecoderLayer"],
    )

    for key in device_map:
        device_map[key] = device

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "model.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder=f"/tmp/offload_gpu{gpu_id}"
    )

    # Manually load connector weights that don't match MLPconnector structure
    from safetensors import safe_open
    with safe_open(os.path.join(model_path, "model.safetensors"), framework="pt", device="cpu") as f:
        if hasattr(model, 'connector') and model.connector is not None:
            connector_state_dict = {}
            for key in f.keys():
                if key.startswith("connector."):
                    new_key = key.replace("connector.", "")
                    connector_state_dict[new_key] = f.get_tensor(key)
            if connector_state_dict:
                new_connector_state = {}
                new_connector_state['fc1.weight'] = connector_state_dict.get('layers.0.weight')
                new_connector_state['fc1.bias'] = connector_state_dict.get('layers.0.bias')
                new_connector_state['fc2.weight'] = connector_state_dict.get('layers.2.weight')
                new_connector_state['fc2.bias'] = connector_state_dict.get('layers.2.bias')
                model.connector.load_state_dict(new_connector_state, strict=True)

    if use_vae_transform:
        vae_transform = VAEImageTransform(vae_max_size, vae_min_size, vae_stride)
    else:
        vae_transform = SigLIPImageTransform(vae_max_size, vae_min_size, vae_stride)
    vit_transform = SigLIPImageTransform(vit_max_size, vit_min_size, vit_stride)

    return model.eval(), vae_model, tokenizer, new_token_ids, vae_transform, vit_transform
