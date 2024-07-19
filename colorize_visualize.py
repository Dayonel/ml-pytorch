import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms.functional import rgb_to_grayscale


def visualize(test_loader, model, device):
    with torch.no_grad():
        # Get the first item
        test_loader_iter = iter(test_loader)
        image, _ = next(test_loader_iter)

        grayscale_images = rgb_to_grayscale(image).to(device)
        colorized_images = model(grayscale_images)

        # Convert the tensors back to CPU for visualization
        grayscale_images_cpu = grayscale_images.cpu().squeeze(1)  # remove the color channel
        colorized_images_cpu = colorized_images.cpu()
        original_images_cpu = image.cpu()

        # Visualize the grayscale, colorized, and original images
        visualize_three_images(original_images_cpu,
                               grayscale_images_cpu, colorized_images_cpu)

# Displays 3 images:
    # black & white
    # colorized
    # original


def visualize_three_images(original_images, grayscale_images, colorized_images, n=3):
    # Image size
    plt.figure(figsize=(6, n*2))

    for i in range(n):
        # Display black & white
        ax = plt.subplot(n, 3, i * 3 + 1)
        show_image(grayscale_images[i])
        ax.set_title("Black & White")
        ax.axis("off")

        # Display colorized
        ax = plt.subplot(n, 3, i * 3 + 2)
        show_image(colorized_images[i])
        ax.set_title("Colorized")
        ax.axis("off")

        # Display original
        ax = plt.subplot(n, 3, i * 3 + 3)
        show_image(original_images[i])
        ax.set_title("Original")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# Displays a single image
def show_image(img):
    # Convert from Tensor image and display
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    if len(img.shape) == 2:  # grayscale image
        plt.imshow(npimg, cmap='gray')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
