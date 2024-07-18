# ML-PyTorch

Colorize black and white images using PyTorch.

## Installation

Install [Python](https://www.python.org/downloads/)

Install datasets and Pillow libraries

```bash
pip install datasets Pillow
```

Install huggingface cli

```bash
pip install -U "huggingface_hub[cli]"
```

Create a virtual environment

```bash
python -m venv .env

# Linux and macOS
source .env/bin/activate

# Windows
.env/Scripts/activate

pip install --upgrade huggingface_hub
```

Create new [token](https://huggingface.co/settings/tokens) with permissions
> Read access to contents of all public gated repos you can access

Login to huggingface and paste token

```bash
huggingface-cli login
```

## Download dataset

We are going to use [imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) dataset.

```bash
python

from datasets import load_dataset
ds = load_dataset("imagenet-1k", trust_remote_code=True)
train_ds = ds["train"]
train_ds[0]["image"]  # a PIL Image
```
