import torch.nn as nn
import torch
# nn module is used to train and build the layers of neural networks

# Colorization neural network
# Transforms grayscale -> color


class ColorizeModel(nn.Module):
    # kernel_size: size of the convolutional filter 5x5
    # stride: filter moves 1 pixel at a time
    # padding: 4 pixels (all 4 sides)
    # dilation: spacing between kernel elements (gap of 1 pixel)
    def __init__(self):
        # We define 4 convolutional layers (2d)
        super(ColorizeModel, self).__init__()

        # takes 1 channel (grayscale) -> produces 64 channels (feature maps)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5,
                               stride=1, padding=4, dilation=2)

        # takes previous 64 channels and produces another 64 channels
        # it learns 64 different filters
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5,
                               stride=1, padding=4, dilation=2)

        # takes previous 64 channels and produces another 128 channels
        # capturing more complex patterns
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5,
                               stride=1, padding=4, dilation=2)

        # condenses this down to 3 channels (rgb)
        self.conv4 = nn.Conv2d(128, 3, kernel_size=5,
                               stride=1, padding=4, dilation=2)

    # this is the data flow through each layer
    # activated by relu function
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))

        # squashes input to a range [0, 1] (normalize)
        x = torch.sigmoid(self.conv4(x))
        return x
