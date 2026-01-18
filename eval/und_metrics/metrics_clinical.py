import os
from .chexbert import CheXbert
import numpy as np

"""
0 = blank/not mentioned
1 = positive
2 = negative
3 = uncertain
"""

CONDITIONS = [
    'enlarged_cardiomediastinum',
    'cardiomegaly',
    'lung_opacity',
    'lung_lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural_effusion',
    'pleural_other',
    'fracture',
    'support_devices',
    'no_finding',
]

class CheXbertMetrics():
    def __init__(self, checkpoint_path, mbatch_size, device):
        self.checkpoint_path = checkpoint_path
        self.mbatch_size = mbatch_size
        self.device = device
        self.chexbert = CheXbert(self.checkpoint_path, self.device,).to(self.device)

    def mini_batch(self, gts, res, mbatch_size=16):
        length = len(gts)
        assert length == len(res)
        for i in range(0, length, mbatch_size):
            batch_gts = gts[i:min(i + mbatch_size, length)]
            batch_res = res[i:min(i + mbatch_size, length)]
            
            # Add length limit logic
            truncated_gts = [text[:512] if len(text) > 512 else text for text in batch_gts]
            truncated_res = [text[:512] if len(text) > 512 else text for text in batch_res]
            
            yield truncated_gts, truncated_res

    def compute(self, gts, res):
        gts_chexbert = []
        res_chexbert = []
        for gt, re in self.mini_batch(gts, res, self.mbatch_size):
            gt_chexbert = self.chexbert(list(gt)).tolist()
            re_chexbert = self.chexbert(list(re)).tolist()
            gts_chexbert += gt_chexbert
            res_chexbert += re_chexbert
        
        gts_chexbert = np.array(gts_chexbert)
        res_chexbert = np.array(res_chexbert)

        # Calculate metrics for both cases
        results = {}

        # Case 1: uncertain as negative (only 1 is considered positive)
        results.update(self._calculate_metrics(
            gts_chexbert == 1,
            res_chexbert == 1,
            prefix="uncertain_as_negative_"
        ))

        # Case 2: uncertain as positive (both 1 and 3 are considered positive)
        results.update(self._calculate_metrics(
            np.logical_or(gts_chexbert == 1, gts_chexbert == 3),
            np.logical_or(res_chexbert == 1, res_chexbert == 3),
            prefix="uncertain_as_positive_"
        ))
        
        return results
    
    def _calculate_metrics(self, gts_bool, res_bool, prefix=""):
        """Calculate various metrics, reusable helper method"""
        tp = (res_bool * gts_bool).astype(float)
        fp = (res_bool * ~gts_bool).astype(float)
        fn = (~res_bool * gts_bool).astype(float)

        # Aggregate by class
        tp_cls = tp.sum(0)
        fp_cls = fp.sum(0)
        fn_cls = fn.sum(0)

        # Aggregate by sample
        tp_eg = tp.sum(1)
        fp_eg = fp.sum(1)
        fn_eg = fn.sum(1)

        # Calculate per-class metrics - add small epsilon to avoid division by zero
        epsilon = 1e-10  # Small value to avoid division by zero
        f1_class = np.nan_to_num(tp_cls / (tp_cls + 0.5 * (fp_cls + fn_cls) + epsilon))

        # Calculate macro-F1 and micro-F1 across all classes
        macro_f1 = f1_class.mean()
        total_tp = tp_cls.sum()
        total_fp = fp_cls.sum()
        total_fn = fn_cls.sum()
        micro_f1 = np.nan_to_num(total_tp / (total_tp + 0.5 * (total_fp + total_fn) + epsilon))

        top5_diseases = ['atelectasis', 'cardiomegaly', 'edema', 'consolidation', 'pleural_effusion']
        top5_indices = [CONDITIONS.index(disease) for disease in top5_diseases if disease in CONDITIONS]

        # Calculate top5 macro-F1 (average F1 across diseases)
        f1_top5_macro = f1_class[top5_indices].mean() if top5_indices else 0.0

        # Calculate top5 micro-F1 (combine TP, FP, FN for all top5 diseases)
        top5_tp = tp_cls[top5_indices].sum() if top5_indices else 0.0
        top5_fp = fp_cls[top5_indices].sum() if top5_indices else 0.0
        top5_fn = fn_cls[top5_indices].sum() if top5_indices else 0.0
        f1_top5_micro = np.nan_to_num(top5_tp / (top5_tp + 0.5 * (top5_fp + top5_fn) + epsilon))

        # Build result dictionary
        scores = {
            # example-based CE metrics
            f'{prefix}ce_num_examples': float(len(res_bool)),
            f'{prefix}ce_precision': float(np.nan_to_num(tp_eg / (tp_eg + fp_eg + epsilon)).mean()),
            f'{prefix}ce_recall': float(np.nan_to_num(tp_eg / (tp_eg + fn_eg + epsilon)).mean()),
            f'{prefix}ce_f1': float(np.nan_to_num(tp_eg / (tp_eg + 0.5 * (fp_eg + fn_eg) + epsilon)).mean()),

            # class-based metrics
            f'{prefix}macro_f1': float(macro_f1),
            f'{prefix}micro_f1': float(micro_f1),
            f'{prefix}f1_top5_macro': float(f1_top5_macro),
            f'{prefix}f1_top5_micro': float(f1_top5_micro)
        }

        # Add F1 score for each class as separate key-value pairs
        for i, condition in enumerate(CONDITIONS):
            scores[f'{prefix}f1_{condition}'] = float(f1_class[i])
            
        return scores

