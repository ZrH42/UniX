"""Common worker utilities for parallel evaluation.

This module provides shared worker process logic used across different
evaluation scripts for distributed inference.

Classes:
    BaseWorker: Base class for evaluation workers
    UnderstandingWorker: Worker for understanding evaluation
    GenerationWorker: Worker for generation evaluation
"""

import os
import time
import torch
import multiprocessing as mp
from multiprocessing import Queue, Process, Lock
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional
from tqdm import tqdm
import queue

from inference import UniXInferencer, setup_random_seed
from eval.base.model_utils import initialize_model_on_gpu, load_config


class BaseWorker(Process):
    """Base class for evaluation workers.

    Attributes:
        gpu_id: GPU device ID
        task_queue: Queue of tasks to process
        result_queue: Queue to put results
        model_path: Path to model directory
        inference_hyper: Inference hyperparameters
    """

    def __init__(
        self,
        gpu_id: int,
        task_queue: Queue,
        result_queue: Queue,
        model_path: str,
        inference_hyper: Dict,
        **kwargs
    ):
        super().__init__()
        self.gpu_id = gpu_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.model_path = model_path
        self.inference_hyper = inference_hyper
        self.kwargs = kwargs

    def run(self):
        """Run the worker process."""
        setup_random_seed(42 + self.gpu_id)
        torch.cuda.set_device(self.gpu_id)

        process_id = os.getpid()
        print(f"Process {process_id} started on GPU {self.gpu_id}")

        self._initialize()
        self._process_tasks()

    @abstractmethod
    def _initialize(self):
        """Initialize model and inferencer. Override in subclass."""
        pass

    @abstractmethod
    def _process_tasks(self):
        """Process tasks from queue. Override in subclass."""
        pass

    @abstractmethod
    def _execute_task(self, task) -> Dict:
        """Execute a single task. Override in subclass."""
        pass


class UnderstandingWorker(BaseWorker):
    """Worker for understanding evaluation.

    Processes image understanding tasks and outputs text reports.
    """

    def __init__(
        self,
        gpu_id: int,
        task_queue: Queue,
        result_queue: Queue,
        model_path: str,
        inference_hyper: Dict,
        output_file: str,
        csv_lock: Lock,
        image_prefix: str = "",
    ):
        super().__init__(gpu_id, task_queue, result_queue, model_path, inference_hyper)
        self.output_file = output_file
        self.csv_lock = csv_lock
        self.image_prefix = image_prefix

    def _initialize(self):
        """Initialize model for understanding tasks."""
        self.model, self.vae_model, self.tokenizer, self.new_token_ids, \
            self.vae_transform, self.vit_transform = initialize_model_on_gpu(
                self.model_path,
                self.gpu_id,
                use_vae_transform=False
            )
        self.inferencer = UniXInferencer(
            model=self.model,
            vae_model=self.vae_model,
            tokenizer=self.tokenizer,
            vae_transform=self.vae_transform,
            vit_transform=self.vit_transform,
            new_token_ids=self.new_token_ids
        )
        torch.cuda.empty_cache()

    def _process_tasks(self):
        """Process understanding tasks from queue."""
        while True:
            try:
                task = self.task_queue.get(timeout=5)
                if task is None:
                    break
                result = self._execute_task(task)
                self._save_result(result)
                self.result_queue.put(result)
            except queue.Empty:
                continue

    def _execute_task(self, task) -> Dict:
        """Execute a single understanding task.

        Args:
            task: Tuple of (idx, user_input, ground_truth, image_paths)

        Returns:
            Result dictionary with prediction and metadata
        """
        idx, user_input, ground_truth, image_paths = task
        user_input = user_input.replace('<image>', '')

        try:
            image = self._load_image(image_paths) if image_paths else None

            torch.cuda.set_device(self.gpu_id)
            start_time = time.time()
            output_dict = self.inferencer(
                image=image,
                text=user_input,
                understanding_output=True,
                **self.inference_hyper
            )
            end_time = time.time()

            predicted_output = output_dict['text']

            return {
                "idx": idx,
                "gpu_id": self.gpu_id,
                "image_path": image_paths[0] if image_paths else "",
                "predicted_output": predicted_output,
                "ground_truth": ground_truth,
                "inference_time": end_time - start_time,
                "success": True
            }
        except Exception as e:
            torch.cuda.empty_cache()
            print(f"Error on GPU {self.gpu_id} task {idx}: {e}")
            return {
                "idx": idx,
                "gpu_id": self.gpu_id,
                "image_path": image_paths[0] if image_paths else "",
                "predicted_output": "",
                "ground_truth": ground_truth,
                "inference_time": 0,
                "success": False,
                "error": str(e)
            }

    def _load_image(self, image_paths: list):
        """Load and preprocess image.

        Args:
            image_paths: List of image paths

        Returns:
            Preprocessed PIL Image
        """
        from PIL import Image
        image = Image.open(image_paths[0]) if image_paths else None
        if image is not None:
            target_size = (384, 384)
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            image.thumbnail(target_size, Image.Resampling.LANCZOS)
            new_image = Image.new('RGBA', target_size, (255, 255, 255, 0))
            x = (target_size[0] - image.size[0]) // 2
            y = (target_size[1] - image.size[1]) // 2
            new_image.paste(image, (x, y), image)
            image = new_image
        return image

    def _save_result(self, result: Dict):
        """Save result to CSV file.

        Args:
            result: Result dictionary
        """
        import csv
        fieldnames = ['Image Path', 'Report Impression', 'Ground Truth']
        csv_result = {
            'Image Path': result.get('image_path', ''),
            'Report Impression': result.get('predicted_output', ''),
            'Ground Truth': result.get('ground_truth', '')
        }

        with self.csv_lock:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            file_exists = os.path.exists(self.output_file)
            with open(self.output_file, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(csv_result)


class GenerationWorker(BaseWorker):
    """Worker for generation evaluation.

    Processes text prompts and generates images.
    """

    def __init__(
        self,
        gpu_id: int,
        task_queue: Queue,
        result_queue: Queue,
        model_path: str,
        inference_hyper: Dict,
        output_dir: str,
    ):
        super().__init__(gpu_id, task_queue, result_queue, model_path, inference_hyper)
        self.output_dir = output_dir

    def _initialize(self):
        """Initialize model for generation tasks."""
        self.model, self.vae_model, self.tokenizer, self.new_token_ids, \
            self.vae_transform, self.vit_transform = initialize_model_on_gpu(
                self.model_path,
                self.gpu_id,
                use_vae_transform=True
            )
        self.inferencer = UniXInferencer(
            model=self.model,
            vae_model=self.vae_model,
            tokenizer=self.tokenizer,
            vae_transform=self.vae_transform,
            vit_transform=self.vit_transform,
            new_token_ids=self.new_token_ids
        )
        torch.cuda.empty_cache()

    def _process_tasks(self):
        """Process generation tasks from queue."""
        while True:
            try:
                task = self.task_queue.get(timeout=5)
                if task is None:
                    break
                result = self._execute_task(task)
                self.result_queue.put(result)
            except queue.Empty:
                continue

    def _execute_task(self, task) -> Dict:
        """Execute a single generation task.

        Args:
            task: Tuple of (idx, prompt, filename)

        Returns:
            Result dictionary with generation status and metadata
        """
        idx, prompt, filename = task

        try:
            torch.cuda.set_device(self.gpu_id)
            start_time = time.time()
            output_dict = self.inferencer(text=prompt, **self.inference_hyper)
            end_time = time.time()

            success = False
            error = None
            if 'image' in output_dict and output_dict['image'] is not None:
                saved_path = self._save_image(output_dict['image'], filename)
                if saved_path:
                    success = True
                else:
                    error = "Save failed"
            else:
                error = "No image generated"

            return {
                "idx": idx,
                "gpu_id": self.gpu_id,
                "success": success,
                "error": error,
                "inference_time": end_time - start_time
            }
        except Exception as e:
            print(f"Error on GPU {self.gpu_id} task {idx}: {e}")
            return {
                "idx": idx,
                "gpu_id": self.gpu_id,
                "success": False,
                "error": str(e),
                "inference_time": 0
            }

    def _save_image(self, image, filename: str) -> Optional[str]:
        """Save generated image to output directory.

        Args:
            image: PIL Image to save
            filename: Output filename

        Returns:
            Path to saved image or None if failed
        """
        from PIL import Image
        os.makedirs(self.output_dir, exist_ok=True)
        if not filename.endswith(('.png', '.jpg', '.jpeg')):
            filename += '.png'
        filepath = os.path.join(self.output_dir, filename)
        try:
            if isinstance(image, Image.Image):
                image.save(filepath)
                return filepath
        except Exception as e:
            print(f"Failed to save image: {e}")
        return None


def run_workers(
    worker_class,
    num_gpus: int,
    task_queue: Queue,
    result_queue: Queue,
    model_path: str,
    inference_hyper: Dict,
    total_tasks: int,
    **worker_kwargs
) -> list:
    """Launch worker processes and collect results.

    Args:
        worker_class: Worker class to instantiate
        num_gpus: Number of GPUs available
        task_queue: Queue of tasks
        result_queue: Queue for results
        model_path: Path to model directory
        inference_hyper: Inference hyperparameters
        total_tasks: Total number of tasks
        **worker_kwargs: Additional arguments for worker

    Returns:
        List of result dictionaries
    """
    processes = []
    for gpu_id in range(num_gpus):
        p = worker_class(
            gpu_id=gpu_id,
            task_queue=task_queue,
            result_queue=result_queue,
            model_path=model_path,
            inference_hyper=inference_hyper,
            **worker_kwargs
        )
        p.start()
        processes.append(p)

    results = []
    with tqdm(total=total_tasks, desc="Processing") as pbar:
        for _ in range(total_tasks):
            result = result_queue.get()
            results.append(result)
            pbar.update(1)

    for _ in range(num_gpus):
        task_queue.put(None)

    for p in processes:
        p.join()

    return results
