import torch
import pathlib
from torch.utils.data import Dataset
from PIL import Image
from colorize_utils import find_classes
from typing import Tuple

# Write a custom dataset class (inherits from torch.utils.data.Dataset)


class ColorizeDataset(Dataset):
    def __init__(self, targ_dir: str, transform=None) -> None:
        # Get all image paths - note: you'd have to update this if you've got .png's or .jpeg's
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_index = find_classes(targ_dir)

    # Load images via path
    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        # expects path in data_folder/class_name/image.jpeg
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        index = self.class_index[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), index
        else:
            return img, index
