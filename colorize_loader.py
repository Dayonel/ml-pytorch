from torch.utils.data import DataLoader
from colorize_dataset import ColorizeDataset
import os

# Turn our images into a Dataset capable of being used with PyTorch


def load(transform, train_dir, test_dir):

    # Use out Custom class to create dataset(s)
    train_dataset = ColorizeDataset(targ_dir=train_dir, transform=transform)
    test_dataset = ColorizeDataset(targ_dir=test_dir, transform=transform)

    # Setup batch size and number of workers
    BATCH_SIZE = 32
    NUM_WORKERS = os.cpu_count()
    print(f"DataLoader batch size: {BATCH_SIZE}, num workers: {NUM_WORKERS}")

    # Turning our Dataset's into DataLoader's makes them iterable so a model can go through learn the relationships between samples and targets (features and labels).
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, test_loader
