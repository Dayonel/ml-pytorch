from typing import Tuple, Dict, List
import os

# # Finds the class folder names in a target directory.
# # Example:
# # find_classes("food_images/train")
# # (["class_1", "class_2"], {"class_1": 0, ...})


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:

    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(
        directory) if entry.is_dir())

    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

    # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_index = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_index
