import torch
from pathlib import Path
from torchvision import transforms
from colorize_loader import load
from colorize_dataset import ColorizeDataset
from colorize_model import ColorizeModel
from colorize_train import train
import torch.nn as nn
import torch.optim as optim

# Protect multi-threading
if __name__ == '__main__':
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

    # Data from images in folders into Tensors
    train_loader, test_loader = load(transform, train_dir, test_dir)

    # Colorization model neural network
    model = ColorizeModel().to(device)

    # mean squared error function (measures the average of the squares of the errors)
    criterion = nn.MSELoss()

    # Adam algorithm with learning rate of 0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # train network
    train(train_loader, device, model, criterion, optimizer)
