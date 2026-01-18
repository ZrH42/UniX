# Installation

## Environment

```bash
git clone https://github.com/ZrH42/UniX.git
cd UniX
conda create -n unix python=3.10 -y
conda activate unix
bash install.sh
```

## Weights


### Download Links

| Model | Download |
|-------|----------|
| UniX | [HuggingFace](https://huggingface.co/ZrH42/UniX) |
| VLM (Janus-Pro-1B) | [HuggingFace](https://huggingface.co/deepseek-ai/Janus-Pro-1B) |
| ViT (SigLIP) | [HuggingFace](https://huggingface.co/google/siglip-large-patch16-384) |
| VAE and VAE config | [Offline ](https://ommer-lab.com/files/latent-diffusion/kl-f16.zip)and [Github](https://github.com/CompVis/latent-diffusion/tree/main/models/first_stage_models/kl-f16) |
| CheXbert (optional) | [GitHub](https://github.com/stanfordmlgroup/CheXbert) |

### Directory Structure

Place all models under `weights/` directory:

```
weights/
├── UniX/
├── Janus-Pro-1B/
├── siglip-large-patch16-384/
├── vae/f16d16/
│         ├── kl-f16d16.ckpt
│         └── kl-f16d16.yaml
└── chexbert/chexbert.pth (optional)
```
