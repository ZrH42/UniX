"""Demo script for UniX inference module.

Usage:
    python demo.py --task understanding --image path/to/image.jpg --prompt "describe this xray"
    python demo.py --task generation --prompt "Heart size is normal"
"""

import os
import time
import random
import argparse
import numpy as np
import torch
from PIL import Image

from inference import setup_model, ModelConfig


def save_image(image, output_dir="output_images", filename=None):
    """Save image to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        timestamp = int(time.time())
        filename = f"generated_{timestamp}.png"
    if not filename.endswith(('.png', '.jpg', '.jpeg')):
        filename += '.png'
    filepath = os.path.join(output_dir, filename)
    image.save(filepath)
    print(f"Image saved to: {filepath}")
    return filepath


def run_understanding(inferencer, image_path, prompt):
    """Run image understanding task."""
    print(f"\n{'='*3} Image Understanding {'='*3}")
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")
    print('-' * 50)

    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return

    image = Image.open(image_path)
    start_time = time.time()

    output = inferencer(image=image, text=prompt, understanding_output=True)

    elapsed = time.time() - start_time
    print(f"Result: {output.get('text', 'No output')}")
    print(f"Inference time: {elapsed:.2f}s")


def run_generation(inferencer, prompt, output_filename=None):
    """Run image generation task."""
    print(f"\n{'='*3} Image Generation {'='*3}")
    print(f"Prompt: {prompt}")
    print('-' * 50)

    start_time = time.time()

    output = inferencer(text=prompt)

    elapsed = time.time() - start_time
    if output.get('image'):
        save_image(output['image'], filename=output_filename)
        print(f"Inference time: {elapsed:.2f}s")
    else:
        print("No image generated")


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # Set random seed
    set_seed(42)

    parser = argparse.ArgumentParser(description="UniX Demo")
    parser.add_argument("--task", type=str, default="understanding",
                        choices=["understanding", "generation"],
                        help="Task type: understanding or generation")
    parser.add_argument("--image", type=str, default="inference/demo_image/xray-p.jpg",
                        help="Path to input image (for understanding)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt text")
    parser.add_argument("--model_path", type=str,
                        default="weights/UniX",
                        help="Path to UniX model (contains all components)")
    parser.add_argument("--vae_path", type=str,
                        default=None,
                        help="Path to VAE checkpoint (optional, defaults to model_path/vae/)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename for generated image")

    args = parser.parse_args()

    # Determine if VAE is needed (only for generation tasks)
    load_vae = (args.task == "generation")

    # Setup model
    config = ModelConfig(
        model_path=args.model_path,
        vae_path=args.vae_path,
    )
    print(f"Loading model... (load_vae={load_vae})")
    inferencer = setup_model(config, load_vae=load_vae)
    print("Model loaded successfully!")

    # Default prompts
    if args.prompt is None:
        if args.task == "understanding":
            args.prompt = "As an imaging expert, review this X-ray and share the FINDINGS and IMPRESSION."
        else:
            args.prompt = "Heart size is normal. Mediastinal contour is unremarkable. No pleural effusion or pneumothorax."

    # Run task
    if args.task == "understanding":
        if args.image is None:
            parser.error("--image is required for understanding task")
        run_understanding(inferencer, args.image, args.prompt)
    else:
        run_generation(inferencer, args.prompt, args.output)

    print(f"\n{'='*3} Demo Completed {'='*3}")


if __name__ == "__main__":
    main()
