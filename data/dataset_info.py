from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    't2i_finetune': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
}


DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': '/data/zhangruiheng/Unix_dataset/t2i', # path of the parquet files
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 237387, # number of total samples in the dataset
        },
    },
    't2i_finetune': {
        't2i': {
            'data_dir': '/data/zhangruiheng/Unix_dataset/t2i_512', # path of the parquet files
            'num_files': 40, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 237387, # number of total samples in the dataset
        },
    },
    'vlm_sft': {
        'llava_ov': {
			'data_dir': '/data/zhangruiheng/Unix_dataset/vlm/images_mimic',
			'jsonl_path': '/data/zhangruiheng/Unix_dataset/vlm/mimic_mmu_si.jsonl',
			'num_total_samples': 163344
		},
    },
}
