import torch
from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Setup train and testing paths
data_path = Path("dataset/")
train_dir = data_path / "train"
test_dir = data_path / "test"

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Write transform for image
data_transform = transforms.Compose([
    # Resize the images to 64x64
    transforms.Resize(size=(64, 64)),
    # Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
])

# Use ImageFolder to create dataset(s)
train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=data_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

test_data = datasets.ImageFolder(root=test_dir, 
                                 transform=data_transform)

print(f"Train data:\n{train_data}\nTest data:\n{test_data}")