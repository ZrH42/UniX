"""Dataset utilities for evaluation.

This module provides dataset classes and utilities for loading and processing
evaluation data for both understanding and generation tasks.

Classes:
    UnderstandingDataset: Dataset for image understanding evaluation
    GenerationDataset: Dataset for text-to-image generation evaluation
"""

import os
import json
import csv
from typing import List, Dict, Any, Optional


class UnderstandingDataset:
    """Dataset for image understanding evaluation.

    Loads JSON data containing image paths, user inputs, and ground truth reports.

    Args:
        data: List of data items or path to JSON file
        image_prefix: Prefix to prepend to image paths
        max_items: Maximum number of items to load (None for all)

    Attributes:
        data: List of processed data items
    """

    def __init__(
        self,
        data: Any,
        image_prefix: str = "",
        max_items: Optional[int] = None
    ):
        if isinstance(data, str):
            if os.path.exists(data):
                with open(data, "r", encoding='utf-8') as f:
                    self.data = json.load(f)
            else:
                raise FileNotFoundError(f"Dataset file not found: {data}")
        else:
            self.data = data

        if max_items:
            self.data = self.data[:max_items]

        self.image_prefix = image_prefix

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single dataset item.

        Args:
            idx: Item index

        Returns:
            Dictionary containing idx, user_input, ground_truth, and images
        """
        item = self.data[idx]
        messages = item["messages"]
        user_input = next(msg["content"] for msg in messages if msg["role"] == "user")
        assistant_output = next(msg["content"] for msg in messages if msg["role"] == "assistant")
        images = item["images"]

        if self.image_prefix:
            images = [os.path.join(self.image_prefix, img) for img in images]

        return {
            "idx": idx,
            "user_input": user_input,
            "ground_truth": assistant_output,
            "images": images
        }


class GenerationDataset:
    """Dataset for text-to-image generation evaluation.

    Loads CSV data containing prompts and filenames.

    Args:
        csv_path: Path to CSV file
        max_items: Maximum number of items to load (None for all)

    Attributes:
        data: List of processed data items
    """

    def __init__(self, csv_path: str, max_items: Optional[int] = None):
        self.data = []

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if 'annotated_prompt' in row and 'synthetic_filename' in row:
                    self.data.append({
                        'prompt': row['annotated_prompt'],
                        'filename': row['synthetic_filename']
                    })

        if max_items:
            self.data = self.data[:max_items]

        print(f"Loaded {len(self.data)} records from CSV.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single dataset item.

        Args:
            idx: Item index

        Returns:
            Dictionary containing idx, prompt, and filename
        """
        item = self.data[idx]
        return {
            "idx": idx,
            "prompt": item["prompt"],
            "filename": item["filename"]
        }


def load_understanding_data(
    dataset_paths: List[str],
    image_prefix: str = "",
    max_items: Optional[int] = None
) -> UnderstandingDataset:
    """Load understanding evaluation data from multiple JSON files.

    Args:
        dataset_paths: List of paths to JSON dataset files
        image_prefix: Prefix for image paths
        max_items: Maximum total items to load

    Returns:
        UnderstandingDataset instance
    """
    all_data = []
    for dataset_path in dataset_paths:
        if os.path.exists(dataset_path):
            with open(dataset_path, "r", encoding='utf-8') as f:
                all_data.extend(json.load(f))
        else:
            print(f"Warning: Dataset path not found: {dataset_path}")

    if not all_data:
        raise ValueError(f"No valid data found in {dataset_paths}")

    return UnderstandingDataset(all_data, image_prefix=image_prefix, max_items=max_items)


def load_generation_data(
    csv_path: str,
    max_items: Optional[int] = None
) -> GenerationDataset:
    """Load generation evaluation data from CSV file.

    Args:
        csv_path: Path to CSV file
        max_items: Maximum items to load

    Returns:
        GenerationDataset instance
    """
    return GenerationDataset(csv_path, max_items=max_items)
