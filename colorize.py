import torch
from dataset_loader import load
from dataset_class import ColorizeDataset
from pathlib import Path
from torchvision import transforms

# Setup train and testing paths
data_path = Path("dataset/")
train_dir = data_path / "train"
test_dir = data_path / "test"

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

 # Create simple transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Turn our images into a Dataset capable of being used with PyTorch
train_loader, test_loader = load(transform, train_dir, test_dir)

# Write a custom dataset class
train_data = ColorizeDataset(targ_dir=train_dir, transform=transform)
test_data = ColorizeDataset(targ_dir=test_dir, transform=transform)
