#!/usr/bin/env python3
"""Medical Image Understanding Evaluation Script.

This script evaluates the UniX model on image understanding tasks by processing
medical images and generating diagnostic reports. Results are saved to a CSV file.

Usage:
    python medical_und_eval.py --dataset_paths test.json --image_prefix ./images --output_file results.csv

Arguments:
    --dataset_paths: Path(s) to dataset JSON files
    --image_prefix: Prefix for image paths
    --output_file: Output CSV file path
    --max_items: Maximum number of items to process (optional)
    --model_path: Path to model directory (default: models/UniX)
"""

import os
import sys
import argparse
import torch
import multiprocessing as mp
from multiprocessing import Queue, Lock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.base.config_utils import load_config as load_eval_config
from eval.base import (
    UnderstandingWorker,
    UnderstandingDataset,
    load_understanding_data,
)
from inference.config import ModelConfig


def parse_args():
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Medical Image Understanding Evaluation"
    )
    parser.add_argument(
        '--dataset_paths',
        type=str,
        nargs='+',
        default=None,
        help="Path(s) to dataset JSON files"
    )
    parser.add_argument(
        '--image_prefix',
        type=str,
        default="./",
        help="Prefix for image paths"
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default="results/understanding_results.csv",
        help="Output CSV file"
    )
    parser.add_argument(
        '--max_items',
        type=int,
        default=None,
        help="Maximum number of items to process"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help="Path to model directory (default: weights/UniX)"
    )
    return parser.parse_args()


def main():
    """Main entry point for understanding evaluation."""
    args = parse_args()

    config = ModelConfig()
    # Override model_path if provided via command line
    if args.model_path is not None:
        config.model_path = args.model_path
    model_path = config.model_path

    # Load inference config from inference/config.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_json = os.path.join(script_dir, "..", "inference", "config.json")
    full_config = load_eval_config(config_json)
    inference_hyper = full_config.get("inference_config", {}).get("understanding", {}).copy()

    # Get dataset paths (use default if not specified)
    if args.dataset_paths is None:
        from eval.data import get_understanding_data_path
        dataset_paths = [get_understanding_data_path()]
    else:
        dataset_paths = args.dataset_paths

    image_prefix = args.image_prefix
    output_file = args.output_file

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs. Running 1 process per GPU.")

    # Remove existing output file
    if os.path.exists(output_file):
        os.remove(output_file)

    # Load dataset
    dataset = load_understanding_data(
        dataset_paths,
        image_prefix=image_prefix,
        max_items=args.max_items
    )

    if len(dataset) == 0:
        print(f"No valid data found in {dataset_paths}. Exiting.")
        return

    task_queue = Queue()
    result_queue = Queue()
    csv_lock = Lock()

    # Populate task queue
    for i in range(len(dataset)):
        item = dataset[i]
        task = (item["idx"], item["user_input"], item["ground_truth"], item["images"])
        task_queue.put(task)

    # Launch workers
    processes = []
    for gpu_id in range(num_gpus):
        p = UnderstandingWorker(
            gpu_id=gpu_id,
            task_queue=task_queue,
            result_queue=result_queue,
            model_path=model_path,
            inference_hyper=inference_hyper,
            output_file=output_file,
            csv_lock=csv_lock,
            image_prefix=image_prefix
        )
        p.start()
        processes.append(p)

    # Collect results
    total_tasks = len(dataset)
    results = []
    from tqdm import tqdm
    with tqdm(total=total_tasks, desc="Inference Progress") as pbar:
        for _ in range(total_tasks):
            result = result_queue.get()
            results.append(result)
            pbar.update(1)

    # Signal workers to stop
    for _ in range(num_gpus):
        task_queue.put(None)

    for p in processes:
        p.join()

    # Summary
    results.sort(key=lambda x: x["idx"])
    successful_tasks = sum(1 for r in results if r["success"])
    print(f"\nInference completed. Success: {successful_tasks}/{total_tasks}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
