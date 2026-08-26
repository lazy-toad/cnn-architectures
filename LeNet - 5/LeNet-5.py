import torch
import torch.nn as nn
import torch.nn.functional as f


class LeNet5Tanh(nn.Module):
    def forward(self, x):
        return 1.759 * torch.tanh((2.0 / 3.0) * x)


class Subsampling(nn.Module):
    def __init__(self, in_channels, activ) -> None:
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.channel_affine = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, groups=in_channels
        )
        self.activ = activ

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.channel_affine(x)
        x = self.activ(x)
        return x


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        activ = LeNet5Tanh()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool1 = Subsampling(in_channels=6, activ=activ)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = Subsampling(in_channels=16, activ=activ)
        self.conv3fc1 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.output = nn.Linear(in_features=84, out_features=num_classes)

        self.activ = activ

    def forward(self, x):

        # get feature map followed by activation map
        x = self.activ(self.conv1(x))
        # pooling
        x = self.pool1(x)

        # feature map to activ map to pooling
        x = self.activ(self.conv2(x))
        x = self.pool2(x)

        # conv/deep layer
        x = self.activ(self.conv3fc1(x))

        # x is currently shape (batch, 120, 1, 1), need to flatten it to (batch,120)
        x = torch.flatten(x, 1)

        x = self.activ(self.fc2(x))

        x = self.output(x)

        return x
