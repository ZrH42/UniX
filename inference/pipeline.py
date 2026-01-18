import os
import torch
from typing import Dict, Any, Optional
from PIL import Image
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch

from .config import ModelConfig, PROJECT_ROOT
from .inferencer import UniXInferencer

# Default paths - All components are bundled in UniX directory
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "weights", "UniX")
DEFAULT_VAE_PATH = os.path.join(PROJECT_ROOT, "weights", "UniX", "vae", "kl-f16d16.ckpt")
DEFAULT_MAX_MEM_PER_GPU = "48GiB"

# Default transform parameters
DEFAULT_VAE_MAX_SIZE = 256
DEFAULT_VAE_MIN_SIZE = 256
DEFAULT_VAE_STRIDE = 16
DEFAULT_VIT_MAX_SIZE = 384
DEFAULT_VIT_MIN_SIZE = 384
DEFAULT_VIT_STRIDE = 16


def setup_model(
    config: Optional[ModelConfig] = None,
    model_path: Optional[str] = None,
    vae_path: Optional[str] = None,
    max_mem_per_gpu: Optional[str] = None,
    vae_max_size: int = DEFAULT_VAE_MAX_SIZE,
    vae_min_size: int = DEFAULT_VAE_MIN_SIZE,
    vae_stride: int = DEFAULT_VAE_STRIDE,
    vit_max_size: int = DEFAULT_VIT_MAX_SIZE,
    vit_min_size: int = DEFAULT_VIT_MIN_SIZE,
    vit_stride: int = DEFAULT_VIT_STRIDE,
    load_vae: bool = True,
) -> UniXInferencer:
    """Load model and return inferencer.

    Args:
        config: ModelConfig object (optional, can use individual params instead)
        model_path: Path to UniX model directory (contains LLM, ViT, tokenizer)
        vae_path: Path to VAE checkpoint (optional, defaults to UniX/vae/)
        max_mem_per_gpu: Maximum memory per GPU
        vae_max_size: VAE transform max size
        vae_min_size: VAE transform min size
        vae_stride: VAE transform stride
        vit_max_size: ViT transform max size
        vit_min_size: ViT transform min size
        vit_stride: ViT transform stride
        load_vae: Whether to load VAE model (set to False for understanding-only tasks)

    Returns:
        UniXInferencer instance ready for inference
    """
    # Merge config with params
    if config is None:
        config = ModelConfig()

    model_path = model_path or config.model_path or DEFAULT_MODEL_PATH
    # All components (LLM, ViT, tokenizer) are bundled in UniX directory
    llm_path = model_path
    vit_path = model_path
    vae_path = vae_path or config.vae_path or DEFAULT_VAE_PATH
    max_mem_per_gpu = max_mem_per_gpu or DEFAULT_MAX_MEM_PER_GPU

    # Import here to avoid circular dependencies
    from data.transforms import SigLIPImageTransform, VAEImageTransform
    from modeling.unix import UniXConfig, UniX, Qwen2ForCausalLM
    from modeling.autoencoder import load_ae
    from modeling.unix_vlm_manager import UnixVLMComponentManager

    print(f"Initializing model from {model_path}...")

    # Initialize UniX components
    unix_manager = UnixVLMComponentManager(llm_path, vit_path)
    llm_config = unix_manager.create_language_model_config()
    vit_model = unix_manager.load_vision_model()

    # Conditionally load VAE (only needed for generation tasks)
    if load_vae:
        vae_model, vae_config = load_ae(local_path=vae_path)
        vae_model = vae_model.to(dtype=torch.bfloat16)
    else:
        vae_model, vae_config = None, None

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

    # Create model
    # Connector weights will be loaded automatically from checkpoint via load_checkpoint_and_dispatch
    model = UniX(Qwen2ForCausalLM(llm_config), vit_model, unix_config)

    # Load tokenizer and token IDs
    tokenizer = unix_manager.load_tokenizer()
    new_token_ids = unix_manager.get_new_token_ids()
    unix_manager.verify_token_ids()

    # Convert model to bfloat16
    model = model.to(dtype=torch.bfloat16)
    for module in model.modules():
        for param in module.parameters(recurse=False):
            if param is not None:
                param.data = param.data.to(dtype=torch.bfloat16)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data = module.bias.data.to(dtype=torch.bfloat16)

    # Set up device map and load checkpoint
    num_gpus = torch.cuda.device_count()
    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(num_gpus)},
        no_split_module_classes=["UniX", "Qwen2MoTDecoderLayer"],
    )
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "model.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder="/tmp/offload"
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
                # Map checkpoint layers to MLPconnector fc1/fc2
                new_connector_state = {}
                new_connector_state['fc1.weight'] = connector_state_dict.get('layers.0.weight')
                new_connector_state['fc1.bias'] = connector_state_dict.get('layers.0.bias')
                new_connector_state['fc2.weight'] = connector_state_dict.get('layers.2.weight')
                new_connector_state['fc2.bias'] = connector_state_dict.get('layers.2.bias')
                model.connector.load_state_dict(new_connector_state, strict=True)
                print("Connector weights loaded successfully")

    # Set model to evaluation mode
    model = model.eval()

    # Create transforms
    vit_transform = SigLIPImageTransform(vit_max_size, vit_min_size, vit_stride)
    # Only create VAE transform if VAE is loaded
    vae_transform = VAEImageTransform(vae_max_size, vae_min_size, vae_stride) if load_vae else None

    print("Model initialization completed")

    return UniXInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
