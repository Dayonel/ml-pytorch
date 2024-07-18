from torch.utils.data import Dataset
import torch
import pathlib
from PIL import Image

# Write a custom dataset class (inherits from torch.utils.data.Dataset)


class ColorizeDataset(Dataset):
    def __init__(self, targ_dir: str, transform=None) -> None:
        # Get all image paths - note: you'd have to update this if you've got .png's or .jpeg's
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
        self.transform = transform

    # Load images via path
    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        # expects path in data_folder/class_name/image.jpeg
        img = self.load_image(index)

        # Transform if necessary
        if self.transform:
            return self.transform(img)
        else:
            return img
