# Inference & Evaluation

## Inference

```bash
# Gradio WebUI
python demo_gradio.py

# CLI
python demo.py --task understanding --image demo/xray-p.jpg
python demo.py --task generation
```

## Understanding Evaluation

```bash
python eval/medical_und_eval.py --image_prefix /path/to/mimic-cxr-jpg
python eval/calculate_und_metrics.py
```

## Generation Evaluation

```bash
python eval/medical_gen_eval.py
cd eval/gen_metrics
bash scripts/image_quality_metrics_memory_saving.sh -p /path/to/UniX -r /path/to/mimic-cxr-jpg
bash scripts/image_quality_metrics_conditional.sh -p /path/to/UniX -r /path/to/mimic-cxr-jpg
```