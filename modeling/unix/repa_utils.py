# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
import timm
from torchvision.transforms import Normalize
from transformers import AutoModel, AutoImageProcessor

# Import REPA components
import sys
import os

# Constants for image preprocessing
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class REPAProjector(nn.Module):
    """REPA projector that maps model features to encoder feature dimensions"""
    
    def __init__(self, hidden_size: int, projector_dim: int, z_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, z_dim),
        )
    
    def forward(self, x):
        return self.mlp(x)


class REPAEncoderManager:
    """Manages loading and preprocessing for REPA visual encoders"""
    
    def __init__(self, enc_type: str, device: torch.device, resolution: int = 256, 
                 hidden_size: int = None, projector_dim: int = 2048):
        self.enc_type = enc_type
        self.device = device
        self.resolution = resolution
        
        # Load encoder based on type
        if enc_type.lower() == 'dinov2':
            self.encoders, self.encoder_types, self.architectures = self._load_dinov2_only(device, resolution)
        elif enc_type.lower() == 'raddino':
            self.encoders, self.encoder_types, self.architectures = self._load_raddino_only(device, resolution)
        else:
            raise ValueError(f"Unsupported encoder type: {enc_type}. Supported types: 'dinov2', 'raddino'")
        
        # Get encoder dimensions
        if enc_type.lower() == 'dinov2':
            self.z_dims = [encoder.embed_dim for encoder in self.encoders]
        elif enc_type.lower() == 'raddino':
            self.z_dims = [encoder[0].config.hidden_size for encoder in self.encoders]
        
        # Initialize projectors if hidden_size is provided
        self.projectors = None
        if hidden_size is not None:
            self.projectors = self._init_projectors(hidden_size, projector_dim)
    
    def _load_dinov2_only(self, device: torch.device, resolution: int = 256):
        """Load only DINOv2 encoder with offline mode"""
        import timm
        import os

        print("Loading DINOv2 encoder...")

        # Load from environment variable
        local_cache_path = os.environ["DINOV2_PATH"]
        if os.path.exists(local_cache_path):
            print(f"Loading DINOv2 from local cache: {local_cache_path}")
            encoder = torch.hub.load(local_cache_path, 'dinov2_vitb14', pretrained=True, source='local')
            print("DINOv2 loaded successfully from local cache")
        else:
            raise FileNotFoundError(f"DINOv2 not found at {local_cache_path}")
        
        # Remove head and add identity
        if hasattr(encoder, 'head'):
            del encoder.head
        encoder.head = torch.nn.Identity()
        
        # Resize positional embeddings for different resolutions
        patch_resolution = 16 * (resolution // 256)
        if hasattr(encoder, 'pos_embed'):
            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [patch_resolution, patch_resolution],
            )
        
        encoder = encoder.to(device=device)
        encoder.eval()
        
        return [encoder], ['dinov2'], ['vit']
    
    def _load_raddino_only(self, device: torch.device, resolution: int = 256):
        """Load only RadDINO encoder"""
        import os

        print("Loading RadDINO encoder...")

        # Load from environment variable
        local_raddino_path = os.environ["RAD_DINO_PATH"]
        try:
            if os.path.exists(local_raddino_path):
                print(f"Loading RadDINO from local path: {local_raddino_path}")
                model = AutoModel.from_pretrained(local_raddino_path, local_files_only=True)
                processor = AutoImageProcessor.from_pretrained(local_raddino_path, local_files_only=True)
                print("RadDINO loaded successfully from local path")
            else:
                raise FileNotFoundError(f"RadDINO not found at {local_raddino_path}")
        except Exception as e:
            print(f"Failed to load RadDINO from local path: {e}")
            raise RuntimeError(f"Could not load RadDINO encoder from {local_raddino_path}: {e}")

        model = model.to(device=device)
        model.eval()

        return [(model, processor)], ['raddino'], ['vit']
    
    def _init_projectors(self, hidden_size: int, projector_dim: int):
        """Initialize projectors and move to GPU with bfloat16"""
        projectors = nn.ModuleList([
            REPAProjector(hidden_size, projector_dim, z_dim) 
            for z_dim in self.z_dims
        ])
        projectors = projectors.to(device=self.device)
        
        return projectors
        
    def preprocess_image(self, image: torch.Tensor, encoder_type: str) -> torch.Tensor:
        """Preprocess image for encoder"""
        resolution = image.shape[-1]
        
        # Convert to float32 to avoid bf16 issues
        image = image.to(dtype=torch.float32)
 
        if encoder_type.lower() == 'dinov2':
            
            # DINOv2 preprocessing
            image = image / 255.
            image = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(image)
            image = F.interpolate(image, 224 * (resolution // 256), mode='bicubic')
            
        elif encoder_type.lower() == 'raddino':
            # RadDINO preprocessing - convert to PIL format for processor
            from PIL import Image
            
            # Convert from tensor to PIL Image format
            if image.dim() == 4:  # Batch of images
                # Convert each image in the batch
                processed_images = []
                for i in range(image.shape[0]):
                    img = image[i]  # (C, H, W)
                    # Resize to 256x256 for RadDINO
                    img_resized = F.interpolate(img.unsqueeze(0), 224 * (resolution // 256), mode='bicubic').squeeze(0)
                    # Convert to PIL Image format (H, W, C) and scale to 0-255
                    img_np = img_resized.permute(1, 2, 0).cpu().numpy()
                    img_np = (img_np * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_np)
                    processed_images.append(img_pil)
                return processed_images
            else:  # Single image
                img = image  # (C, H, W)
                # Resize to 256x256 for RadDINO
                img_resized = F.interpolate(img.unsqueeze(0), 224 * (resolution // 256), mode='bicubic').squeeze(0)
                img_np = img_resized.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                return [img_pil]
        
        return image
    
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> List[torch.Tensor]:
        """Extract features from encoder for given images following original REPA"""
        import torch.distributed as dist
        
        features = []
        
        for encoder, encoder_type, arch in zip(self.encoders, self.encoder_types, self.architectures):
            if encoder_type.lower() == 'dinov2':
                # Preprocess image for DINOv2
                processed_image = self.preprocess_image(images, encoder_type)
                
                # Extract features - following original REPA implementation
                with torch.no_grad():
                    z = encoder.forward_features(processed_image)
                    z = z['x_norm_patchtokens']
                           
                    features.append(z)
                    
            elif encoder_type.lower() == 'raddino':
                # Preprocess image for RadDINO
                processed_images = self.preprocess_image(images, encoder_type)
                
                # Extract features using RadDINO processor and model
                with torch.no_grad():
                    # Process images through RadDINO processor
                    inputs = encoder[1](images=processed_images, return_tensors="pt")
                    inputs = inputs.to(encoder[0].device)

                    # Get features from RadDINO model
                    outputs = encoder[0](**inputs)
                    
                    z = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
                    z_patches = z[:, 1:, :]  # Remove first token (CLS), keep patch tokens
                    
                    features.append(z_patches)
        
        return features


class REPALoss(nn.Module):
    """REPA projection loss for aligning model features with encoder features"""
    
    def __init__(self, encoders: List[nn.Module], encoder_types: List[str], 
                 z_dims: List[int], projectors: nn.ModuleList):
        super().__init__()
        self.encoders = encoders
        self.encoder_types = encoder_types
        self.z_dims = z_dims
        self.projectors = projectors
    
    def mean_flat(self, x):
        """Take the mean over all non-batch dimensions"""
        return torch.mean(x, dim=list(range(1, len(x.size()))))
    
    def forward(self, model_features: torch.Tensor, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute REPA projection loss following original implementation
        
        Args:
            model_features: Features from the model at encoder_depth layer (N, T, D)
            encoder_features: List of features from pretrained encoders
        Returns:
            projection_loss: Scalar loss value
        """
        import torch.distributed as dist
        
        proj_loss = 0.
        bsz = encoder_features[0].shape[0]
        
        # Project model features to encoder dimensions
        projected_features = []
        for i, projector in enumerate(self.projectors):
            # Reshape model features to (N*T, D) for projection
            N, T, D = model_features.shape
            flat_features = model_features.reshape(-1, D)
            
            # Keep the same dtype as model features (bf16)
            projected = projector(flat_features)
            
            # Reshape back to (N, T, z_dim)
            projected = projected.reshape(N, T, -1)
            projected_features.append(projected)
            
        # Compute alignment loss for single encoder
        z = encoder_features[0]  # (bs, tokennum_enc, embeddim_enc)
        z_tilde = projected_features[0]  # (bs, tokennum_model, embeddim_enc)
        
        # Normalize features
        z_tilde_norm = F.normalize(z_tilde, dim=-1)
        z_norm = F.normalize(z, dim=-1)
        
        # Handle different sequence lengths by interpolation if needed
        if z.shape[1] != z_tilde.shape[1]:
            # Interpolate z_tilde to match z's sequence length
            z_tilde_norm = F.interpolate(
                z_tilde_norm.transpose(1, 2),  # (bs, embeddim_enc, tokennum_model)
                size=z.shape[1],  # tokennum_enc
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # (bs, tokennum_enc, embeddim_enc)
        
        # Compute cosine similarity loss
        cosine_sim = (z_norm * z_tilde_norm).sum(dim=-1)  # (bs, tokennum)
        proj_loss = self.mean_flat(-cosine_sim)  # Negative cosine similarity
        
        # For single encoder, just divide by batch size
        proj_loss /= bsz
        
        # Ensure proj_loss is a scalar tensor
        if proj_loss.dim() > 0:
            proj_loss = proj_loss.mean()
        
        return proj_loss


def create_repa_components(enc_type: str, device: torch.device, 
                          hidden_size: int, resolution: int = 256) -> Tuple[REPAEncoderManager, REPALoss]:
    """Create REPA encoder manager and loss function"""
    
    # Create encoder manager
    encoder_manager = REPAEncoderManager(enc_type, device, resolution, hidden_size)
    
    # Create loss function
    repa_loss = REPALoss(
        encoders=encoder_manager.encoders,
        encoder_types=encoder_manager.encoder_types,
        z_dims=encoder_manager.z_dims,
        projectors=encoder_manager.projectors
    ).to(device)
    
    return encoder_manager, repa_loss