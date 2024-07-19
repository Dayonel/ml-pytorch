# ML-PyTorch

Colorize black and white images using PyTorch.

## Installation

Install [Python](https://www.python.org/downloads/)

Install required libraries

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2.0" matplotlib
```

## Download dataset

We are going to use a subset of [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) dataset: `pizza, steak and sushi`.

[Download it here (15MB)](https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip)

```
pizza_steak_sushi/
   train/
       pizza/
           image01.jpeg
           ...
       steak/
           image04.jpeg
           ...
       sushi/
           image07.jpeg
           ...
   test/
       pizza/
           image101.jpeg
           ...
       steak/
           image104.jpeg
           ...
       sushi/
           image107.jpeg
           ...
```

```bash
# Unzipping archive
tar -xzvf pizza_steak_sushi.zip --directory dataset

# Deleting non-needed anymore archive
rm pizza_steak_sushi.zip
```

## Run

```bash
python colorize.py
```

# Result

![colorized_2](https://github.com/user-attachments/assets/2c9ff2bc-361c-4ef8-b789-3d0c945c16d7)
