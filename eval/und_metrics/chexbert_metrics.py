import torch
import numpy as np
from tqdm import tqdm
from .metrics_clinical import CheXbertMetrics

def calculate_chexbert_metrics(predictions, references, chexbert_path, batch_size=16):
    """Calculate all CheXbert clinical metrics"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize CheXbert evaluator
    chexbert_metrics = CheXbertMetrics(chexbert_path, batch_size, device)

    # Use CheXbert to calculate metrics with progress bar
    # Calculate total batches for progress display
    total_batches = (len(references) + batch_size - 1) // batch_size

    gts_chexbert = []
    res_chexbert = []

    # Use tqdm to wrap mini_batch iteration
    for gt, re in tqdm(
        chexbert_metrics.mini_batch(references, predictions, batch_size),
        total=total_batches,
        desc="Computing CheXbert metrics"
    ):
        gt_chexbert = chexbert_metrics.chexbert(list(gt)).tolist()
        re_chexbert = chexbert_metrics.chexbert(list(re)).tolist()
        gts_chexbert += gt_chexbert
        res_chexbert += re_chexbert

    # Ensure numpy arrays for boolean operations
    gts_chexbert = np.array(gts_chexbert)
    res_chexbert = np.array(res_chexbert)

    # Use numpy arrays for boolean operations
    clinical_scores = chexbert_metrics._calculate_metrics(
        gts_chexbert == 1,
        res_chexbert == 1,
        prefix="uncertain_as_negative_"
    )

    # Also use numpy arrays for uncertain_as_positive case
    uncertain_as_positive_gt = np.logical_or(gts_chexbert == 1, gts_chexbert == 3)
    uncertain_as_positive_pred = np.logical_or(res_chexbert == 1, res_chexbert == 3)
    
    clinical_scores.update(chexbert_metrics._calculate_metrics(
        uncertain_as_positive_gt,
        uncertain_as_positive_pred,
        prefix="uncertain_as_positive_"
    ))
    
    return clinical_scores 