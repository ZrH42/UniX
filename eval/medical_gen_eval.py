#!/usr/bin/env python3
"""Medical Image Generation Evaluation Script.

This script evaluates the UniX model on text-to-image generation tasks by
processing prompts and generating medical images. Generated images are saved
to an output directory.

Usage:
    python medical_gen_eval.py --csv_path prompts.csv --output_dir ./output --cfg_text_scale 2.0

Arguments:
    --csv_path: Path to CSV file with prompts
    --output_dir: Output directory for generated images
    --max_items: Maximum number of items to process (optional)
    --cfg_text_scale: Classifier-free guidance scale for text
    --cfg_img_scale: Classifier-free guidance scale for image
"""

import os
import sys
import argparse
import torch
import multiprocessing as mp
from multiprocessing import Queue

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.config import ModelConfig
from eval.base import (
    load_config as load_eval_config,
    GenerationWorker,
    GenerationDataset,
    load_generation_data,
)


def parse_args():
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Medical Image Generation Evaluation"
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        default=None,
        help="Path to CSV file with prompts"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="results/generation_images",
        help="Output directory for generated images"
    )
    parser.add_argument(
        '--max_items',
        type=int,
        default=None,
        help="Maximum number of items to process"
    )
    parser.add_argument(
        '--cfg_text_scale',
        type=float,
        default=2.0,
        help="Classifier-free guidance scale for text"
    )
    parser.add_argument(
        '--cfg_img_scale',
        type=float,
        default=1.0,
        help="Classifier-free guidance scale for image"
    )
    return parser.parse_args()


def main():
    """Main entry point for generation evaluation."""
    args = parse_args()

    config = ModelConfig()
    model_path = config.model_path

    # Load inference config from inference/config.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_json = os.path.join(script_dir, "..", "inference", "config.json")
    full_config = load_eval_config(config_json)
    inference_hyper = full_config.get("inference_config", {}).get("generation", {}).copy()
    inference_hyper.update({
        "cfg_text_scale": args.cfg_text_scale,
        "cfg_img_scale": args.cfg_img_scale
    })

    # Get CSV path (use default if not specified)
    if args.csv_path is None:
        from eval.data import get_generation_data_path
        csv_path = get_generation_data_path()
    else:
        csv_path = args.csv_path

    # Create output directory with cfg scale suffix
    output_dir = f"{args.output_dir}_c{args.cfg_text_scale}"
    os.makedirs(output_dir, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs. Running 1 process per GPU.")

    # Load dataset
    dataset = load_generation_data(csv_path, max_items=args.max_items)

    if len(dataset) == 0:
        print(f"No valid data found in {csv_path}. Exiting.")
        return

    task_queue = Queue()
    result_queue = Queue()

    # Populate task queue
    for i in range(len(dataset)):
        item = dataset[i]
        task = (item["idx"], item["prompt"], item["filename"])
        task_queue.put(task)

    # Launch workers
    processes = []
    for gpu_id in range(num_gpus):
        p = GenerationWorker(
            gpu_id=gpu_id,
            task_queue=task_queue,
            result_queue=result_queue,
            model_path=model_path,
            inference_hyper=inference_hyper,
            output_dir=output_dir
        )
        p.start()
        processes.append(p)

    # Collect results
    total_tasks = len(dataset)
    from tqdm import tqdm
    with tqdm(total=total_tasks, desc="Generation Progress") as pbar:
        for _ in range(total_tasks):
            result_queue.get()
            pbar.update(1)

    # Signal workers to stop
    for _ in range(num_gpus):
        task_queue.put(None)

    for p in processes:
        p.join()

    print(f"\nGeneration completed. Results in {output_dir}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
