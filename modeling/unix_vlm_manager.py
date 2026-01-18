"""
UniX VLM Component Manager for loading and integrating UniX model components with UniX.
"""

import os
import json
import time
import torch
from collections.abc import Mapping as _Mapping

from .unix_vlm.models import VLChatProcessor
from .unix_vlm.models.clip_encoder import CLIPVisionTower
from .unix_vlm.models.projector import MlpProjector
from .unix import Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig


class AttrDict(_Mapping):
    """A dictionary that allows attribute-style access (Python 3.10 compatible replacement for attrdict)."""

    def __init__(self, *args, **kwargs):
        self._data = dict(*args, **kwargs)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"AttrDict({self._data})"

    def __getattr__(self, key):
        # Handle special attributes first to avoid deepcopy recursion
        if key.startswith('_'):
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key == '_data':
            super().__setattr__(key, value)
        else:
            self._data[key] = value

    def get(self, key, default=None):
        return self._data.get(key, default)


class UnixVLMComponentManager:
    """Manages loading and integration of UniX VLM components with UniX model"""
    
    def __init__(self, llm_path: str, vit_path: str, rank: int = 0, logger=None):
        self.llm_path = llm_path
        self.vit_path = vit_path
        self.rank = rank
        self.logger = logger
        
        # Component storage
        self.language_model = None
        self.vision_model = None
        self.aligner = None
        self.tokenizer = None
        
        # UniX VLM configuration
        self.unix_vlm_config = None
        self._load_unix_vlm_config()
    
    def _load_unix_vlm_config(self):
        """Load UniX VLM model configuration"""
        config_path = os.path.join(self.llm_path, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"UniX VLM config not found at {config_path}")

        with open(config_path, 'r') as f:
            self.unix_vlm_config = json.load(f)
        
    
    def _load_state_dict_from_checkpoint(self, checkpoint_path: str):
        """Load state dict from checkpoint file (supports both .bin and .safetensors)"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        if checkpoint_path.endswith('.safetensors'):
            from safetensors import safe_open
            state_dict = {}
            with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        return state_dict
    
    def _find_checkpoint_file(self, directory: str):
        """Find checkpoint file in directory (prioritizes .safetensors)"""
        weight_files = []
        if os.path.exists(os.path.join(directory, "pytorch_model.bin")):
            weight_files.append(os.path.join(directory, "pytorch_model.bin"))
        if os.path.exists(os.path.join(directory, "model.safetensors")):
            weight_files.append(os.path.join(directory, "model.safetensors"))
        
        if not weight_files:
            raise FileNotFoundError(f"No weight files found in {directory}")
        
        return weight_files[0]  # Return first available (prioritizes .safetensors)
    
    def create_language_model_config(self, model_args=None, training_args=None):
        """Create Qwen2Config from UniX VLM configuration
        
        Args:
            model_args: Optional model arguments for training mode
            training_args: Optional training arguments for training mode
            
        Returns:
            Qwen2Config with appropriate settings
        """
        # If called without arguments (inference mode), use simple configuration
        if model_args is None or training_args is None:
            language_config = self.unix_vlm_config.get("language_config", {})
            
            llm_config = Qwen2Config(
                hidden_size=language_config.get("hidden_size", 2048),
                intermediate_size=language_config.get("intermediate_size", 5632),
                num_hidden_layers=language_config.get("num_hidden_layers", 24),
                num_attention_heads=language_config.get("num_attention_heads", 16),
                num_key_value_heads=language_config.get("num_key_value_heads", 16),
                vocab_size=language_config.get("vocab_size", 102400),
                max_position_embeddings=language_config.get("max_position_embeddings", 16384),
                model_type="llama"
            )
            
            # Apply default UniX settings for inference
            llm_config.layer_module = "Qwen2MoTDecoderLayer"
            llm_config.qk_norm = True
            llm_config.tie_word_embeddings = False
            
            return llm_config
        
        # Training mode with full arguments
        if training_args.finetune_from_hf:
            return Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
        
        language_config = self.unix_vlm_config.get("language_config", {})
        
        llm_config = Qwen2Config(
            hidden_size=language_config.get("hidden_size", 2048),
            intermediate_size=language_config.get("intermediate_size", 5632),
            num_hidden_layers=language_config.get("num_hidden_layers", 24),
            num_attention_heads=language_config.get("num_attention_heads", 16),
            num_key_value_heads=language_config.get("num_key_value_heads", 16),
            vocab_size=language_config.get("vocab_size", 102400),
            max_position_embeddings=language_config.get("max_position_embeddings", 16384),
            model_type="llama"
        )
        
        # Apply UniX-specific settings
        llm_config.layer_module = model_args.layer_module
        llm_config.qk_norm = model_args.llm_qk_norm
        llm_config.tie_word_embeddings = model_args.tie_word_embeddings
        llm_config.freeze_und = training_args.freeze_und
        
        # Apply REPA settings
        llm_config.use_repa = training_args.use_repa
        llm_config.repa_enc_type = training_args.repa_enc_type
        llm_config.repa_projector_dim = training_args.repa_projector_dim
        llm_config.repa_encoder_depth = training_args.repa_encoder_depth
        
        return llm_config
    
    def load_language_model(self, llm_config, training_args):
        """Load and initialize language model with UniX VLM weights"""
        self.language_model = Qwen2ForCausalLM(llm_config)
        self.language_model = self.language_model.to(dtype=torch.bfloat16)
        
        if not training_args.finetune_from_hf:
            self._load_language_model_weights()
        
        if training_args.copy_init_moe:
            self.language_model.init_moe()
        
        return self.language_model
    
    def _load_language_model_weights(self):
        """Load UniX VLM language model weights"""
        if self.rank == 0 and self.logger:
            self.logger.info(f"Loading UniX VLM language model weights from {self.llm_path}")
        
        weight_file = self._find_checkpoint_file(self.llm_path)
        if self.rank == 0 and self.logger:
            self.logger.info(f"Loading weights from: {weight_file}")
        
        state_dict = self._load_state_dict_from_checkpoint(weight_file)
        
        # Extract language model weights
        language_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("language_model."):
                new_key = key.replace("language_model.", "")
                language_state_dict[new_key] = value
        
        if not language_state_dict:
            if self.rank == 0 and self.logger:
                self.logger.warning("No language model weights found in checkpoint")
            return False
        
        if self.rank == 0 and self.logger:
            self.logger.info(f"Successfully loaded {len(language_state_dict)} language model weights")
            self.logger.info(f"Missing {len(self.language_model.state_dict()) - len(language_state_dict)} weights (mostly MoE and bias parameters, which is expected)")
        
        # Load weights into the model
        incompatible_keys = self.language_model.load_state_dict(language_state_dict, strict=False)
        
        if self.rank == 0 and self.logger:
            self.logger.info(f"Successfully loaded language model weights")
            self.logger.info(f"Missing keys: {len(incompatible_keys.missing_keys)}")
            self.logger.info(f"Unexpected keys: {len(incompatible_keys.unexpected_keys)}")
        
        return True
    
    def load_vision_model(self):
        """Load UniX VLM vision model with UniX compatibility"""
        if self.rank == 0 and self.logger:
            self.logger.info("Creating UniX VLM CLIPVisionTower with UniX compatibility")
        
        # Use UniX VLM's exact configuration
        vit_model = CLIPVisionTower(
            model_name="siglip_large_patch16_384",
            image_size=384,
            select_feature="same",
            select_layer=-1
        )
        
        # Load weights
        self._load_vision_model_weights(vit_model)
        
        if self.rank == 0 and self.logger:
            self.logger.info("UniX VLM CLIPVisionTower with UniX compatibility created successfully")
        
        return vit_model
    
    def _load_vision_model_weights(self, vit_model):
        """Load vision model weights from UniX VLM checkpoint"""
        weight_file = self._find_checkpoint_file(self.llm_path)
        
        if self.rank == 0 and self.logger:
            self.logger.info(f"Loading vision model weights from: {weight_file}")
        
        state_dict = self._load_state_dict_from_checkpoint(weight_file)
        
        # Extract vision model weights, excluding gen_* keys
        vision_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("vision_model.") and not key.startswith("gen_"):
                new_key = key.replace("vision_model.", "")
                vision_state_dict[new_key] = value
        
        if vision_state_dict:
            if self.rank == 0 and self.logger:
                self.logger.info(f"Found {len(vision_state_dict)} vision model weights")
            
            # Load weights into the original UniX VLM model (not the wrapper)
            incompatible_keys = vit_model.load_state_dict(vision_state_dict, strict=False)
            
            if self.rank == 0 and self.logger:
                self.logger.info(f"Loaded vision model from {weight_file}")
                
                if incompatible_keys.missing_keys:
                    self.logger.info(f"Missing keys: {len(incompatible_keys.missing_keys)}")
                if incompatible_keys.unexpected_keys:
                    self.logger.info(f"Unexpected keys: {len(incompatible_keys.unexpected_keys)}")
                
                if not incompatible_keys.missing_keys and not incompatible_keys.unexpected_keys:
                    self.logger.info("✓ All vision model weights loaded successfully")
        else:
            if self.rank == 0 and self.logger:
                self.logger.warning(f"No vision model weights found in {weight_file}")
    
    def load_aligner(self):
        """Load UniX VLM aligner (projector)"""
        if self.rank == 0 and self.logger:
            self.logger.info(f"Loading UniX VLM aligner from {self.llm_path}")
        
        aligner_config = self.unix_vlm_config.get("aligner_config", {})
        aligner_params = aligner_config.get("params", {})

        aligner_params = AttrDict(aligner_params)
        
        # Create aligner with UniX VLM configuration
        self.aligner = MlpProjector(aligner_params)
        
        # Load weights
        weight_file = self._find_checkpoint_file(self.llm_path)
        state_dict = self._load_state_dict_from_checkpoint(weight_file)
        
        # Extract aligner weights, excluding gen_* keys
        aligner_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("aligner.") and not key.startswith("gen_"):
                new_key = key.replace("aligner.", "")
                aligner_state_dict[new_key] = value
        
        if not aligner_state_dict:
            raise ValueError(f"No aligner weights found in {weight_file}")
        
        # Load weights into the aligner
        incompatible_keys = self.aligner.load_state_dict(aligner_state_dict, strict=False)
        
        if self.rank == 0 and self.logger:
            self.logger.info(f"Loaded {len(aligner_state_dict)} aligner weights from {weight_file}")
            if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
                self.logger.info(f"Warning: {len(incompatible_keys.missing_keys)} missing, {len(incompatible_keys.unexpected_keys)} unexpected keys")
        
        return self.aligner
    
    def load_tokenizer(self):
        """Load tokenizer using UniX VLM VLChatProcessor"""
        if self.rank == 0 and self.logger:
            self.logger.info("Loading tokenizer using UniX VLM VLChatProcessor...")
        
        vl_chat_processor = VLChatProcessor.from_pretrained(self.llm_path)
        self.tokenizer = vl_chat_processor.tokenizer
        
        if self.rank == 0 and self.logger:
            self.logger.info("Successfully loaded tokenizer using UniX VLM VLChatProcessor")
            self.logger.info(f"Tokenizer vocab size: {len(self.tokenizer)}")
        
        return self.tokenizer
    
    def get_new_token_ids(self):
        """Get UniX VLM special token IDs (pre-defined in UniX VLM tokenizer)

        These token IDs are fixed in UniX VLM pre-training and cannot be changed.
        They correspond to:
        - 100000: <|im_start|> - dialogue start
        - 100001: <|im_end|> - dialogue end  
        - 100003: <|vision_start|> - image start
        - 100580: <|vision_end|> - image end
        """
        return {
            'bos_token_id': 100000,
            'eos_token_id': 100001,
            'start_of_image': 100003,
            'end_of_image': 100580,
        }
    
    def verify_token_ids(self):
        """Verify that UniX VLM token IDs are valid and log them"""
        if self.rank == 0 and self.logger:
            self.logger.info("Verifying UniX VLM token IDs...")
        
        token_info = {
            100000: '<|im_start|>',
            100001: '<|im_end|>',
            100003: '<|vision_start|>',
            100580: '<|vision_end|>',
        }
        
        all_valid = True
        for token_id, token_name in token_info.items():
            if token_id >= len(self.tokenizer):
                if self.rank == 0 and self.logger:
                    self.logger.error(f"  ✗ Token ID {token_id} ({token_name}) is out of range!")
                all_valid = False
            else:
                decoded_token = self.tokenizer.decode([token_id])
                if self.rank == 0 and self.logger:
                    self.logger.info(f"  ✓ ID {token_id} ({token_name}) -> '{decoded_token}'")
        
        if self.rank == 0 and self.logger:
            if all_valid:
                self.logger.info("  ✓ All token IDs are valid")
            else:
                self.logger.error("  ✗ Some token IDs are invalid!")
        
        return all_valid
    
    def create_dummy_vit_config(self, vit_patch_size=16):
        """Create a dummy ViT config for UniX compatibility
        
        Args:
            vit_patch_size: Patch size for the vision transformer (default: 16)
            
        Returns:
            SiglipVisionConfig with UniX VLM vision model specifications
        """
        return SiglipVisionConfig(
            patch_size=vit_patch_size,
            hidden_size=1024,  # UniX VLM vision model hidden size
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            image_size=384,
        )
    
    def replace_model_connector(self, model):
        """Replace model connector with UniX VLM aligner"""
        if self.rank == 0 and self.logger:
            self.logger.info("Loading UniX VLM aligner...")

        unix_vlm_aligner = self.load_aligner()

        if self.rank == 0 and self.logger:
            self.logger.info(f"Successfully loaded unix_vlm_aligner: {type(unix_vlm_aligner)}")
            self.logger.info(f"unix_vlm_aligner parameters count: {sum(p.numel() for p in unix_vlm_aligner.parameters())}")

            # Replace the connector with UniX VLM aligner
            self.logger.info("Replacing model.connector with unix_vlm_aligner...")

            old_connector = model.connector
            self.logger.info(f"Old connector type: {type(old_connector)}")

        model.connector = unix_vlm_aligner

        if self.rank == 0 and self.logger:
            self.logger.info(f"New connector type: {type(model.connector)}")
            self.logger.info("✓ Successfully replaced connector with UniX VLM aligner")
        
        return model
    
    def save_model_weights_info(self, model, training_args, vae_model=None, vae_config=None):
        """Save simplified model weights information to file"""
        if self.rank != 0:
            return
        
        if not self.logger:
            return
            
        self.logger.info("Saving simplified model weights information...")
        weights_info_file = os.path.join(training_args.results_dir, "model_weights_info.txt")
        
        with open(weights_info_file, 'w', encoding='utf-8') as f:
            f.write("Model Weights Information\n")
            f.write("=" * 50 + "\n")
            f.write(f"File Path: {weights_info_file}\n")
            f.write(f"Generation Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Type: UniX with UniX VLM components\n")
            f.write(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
            f.write(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
            f.write(f"Frozen Parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}\n\n")
            
            f.write("Weight Names and Dimensions:\n")
            f.write("-" * 50 + "\n")
            
            # Collect all parameters
            all_params = list(model.named_parameters())
            if vae_model is not None:
                all_params.extend([(f"vae.{n}", p) for n, p in vae_model.named_parameters()])
            
            # Output all weights in order
            for i, (name, param) in enumerate(all_params, 1):
                # Calculate parameter count
                param_num = param.numel()

                # Format shape string
                if len(param.shape) == 1:
                    shape_str = str(param.shape[0])
                elif len(param.shape) == 2:
                    shape_str = f"{param.shape[0]} x {param.shape[1]}"
                elif len(param.shape) == 3:
                    shape_str = f"{param.shape[0]} x {param.shape[1]} x {param.shape[2]}"
                elif len(param.shape) == 4:
                    shape_str = f"{param.shape[0]} x {param.shape[1]} x {param.shape[2]} x {param.shape[3]}"
                else:
                    shape_str = str(list(param.shape))

                # Output format: No. weight_name shape trainable
                f.write(f"{i:3d}. {name:<100} {shape_str:<20} {'Trainable' if param.requires_grad else 'Frozen'}\n")

            f.write("-" * 50 + "\n")
            f.write(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
            f.write(f"Analysis Complete!\n")

        self.logger.info(f"Simplified model weights information saved to: {weights_info_file}")

