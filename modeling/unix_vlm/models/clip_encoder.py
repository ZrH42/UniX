from typing import Dict, List, Literal, Optional, Tuple, Union
import os
import time

import torch
import torch.nn as nn
import torchvision.transforms
from einops import rearrange
import PIL.Image
import torchvision.utils

from .siglip_vit import create_siglip_vit


class CLIPVisionTower(nn.Module):
    def __init__(
        self,
        model_name: str = "siglip_large_patch16_384",
        image_size: Union[Tuple[int, int], int] = 336,
        select_feature: str = "patch",
        select_layer: int = -2,
        select_layers: list = None,
        ckpt_path: str = "",
        pixel_mean: Optional[List[float]] = None,
        pixel_std: Optional[List[float]] = None,
        save_images: bool = False,
        save_dir: str = "./saved_images",
        **kwargs,
    ):
        super().__init__()

        self.model_name = model_name
        self.select_feature = select_feature
        self.select_layer = select_layer
        self.select_layers = select_layers
        self.save_images = save_images
        self.save_dir = save_dir
        self.image_counter = 0
        
        # Create save directory if saving is enabled
        if self.save_images:
            os.makedirs(self.save_dir, exist_ok=True)

        vision_tower_params = {
            "model_name": model_name,
            "image_size": image_size,
            "ckpt_path": ckpt_path,
            "select_layer": select_layer,
        }
        vision_tower_params.update(kwargs)
        self.vision_tower, self.forward_kwargs = self.build_vision_tower(
            vision_tower_params
        )

        if pixel_mean is not None and pixel_std is not None:
            image_norm = torchvision.transforms.Normalize(
                mean=pixel_mean, std=pixel_std
            )
        else:
            image_norm = None

        self.image_norm = image_norm

    def build_vision_tower(self, vision_tower_params):
        if self.model_name.startswith("siglip"):
            self.select_feature = "same"
            vision_tower = create_siglip_vit(**vision_tower_params)
            forward_kwargs = dict()

        elif self.model_name.startswith("sam"):
            vision_tower = create_sam_vit(**vision_tower_params)
            forward_kwargs = dict()

        else:  # huggingface
            from transformers import CLIPVisionModel

            vision_tower = CLIPVisionModel.from_pretrained(**vision_tower_params)
            forward_kwargs = dict(output_hidden_states=True)

        return vision_tower, forward_kwargs

    def feature_select(self, image_forward_outs):
        if isinstance(image_forward_outs, torch.Tensor):
            # the output has been the self.select_layer"s features
            image_features = image_forward_outs
        else:
            image_features = image_forward_outs.hidden_states[self.select_layer]

        if self.select_feature == "patch":
            # if the output has cls_token
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        elif self.select_feature == "same":
            image_features = image_features

        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def forward(self, images=None):
        # Save images if enabled
        if self.save_images and images is not None:
            self._save_images(images)
        
        image_forward_outs = self.vision_tower(images, **self.forward_kwargs)
        image_features = self.feature_select(image_forward_outs)
        return image_features

    def _save_images(self, images):
        """Save images to disk for debugging/visualization purposes"""
        if images is None:
            return
            
        # Convert to CPU and denormalize if needed
        images_cpu = images.detach().cpu()
        
        mean = torch.tensor([0.555895, 0.555895, 0.555895]).view(1, 3, 1, 1)
        std = torch.tensor([0.336385, 0.336385, 0.336385]).view(1, 3, 1, 1)
        images_cpu = images_cpu * std + mean
        
        # Clamp values to [0, 1] range
        images_cpu = torch.clamp(images_cpu, 0, 1)
        
        # Save each image in the batch
        batch_size = images_cpu.shape[0]
        for i in range(batch_size):
            # Create filename with timestamp and counter
            timestamp = int(time.time() * 1000)  # milliseconds
            filename = f"image_{self.image_counter:06d}_{timestamp}_{i}.png"
            filepath = os.path.join(self.save_dir, filename)
            
            # Save the image
            torchvision.utils.save_image(images_cpu[i], filepath)
            
        self.image_counter += batch_size
        
        print(f"Saved {batch_size} images to {self.save_dir} (total saved: {self.image_counter})")

