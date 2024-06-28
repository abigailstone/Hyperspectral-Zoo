import math

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    https://onlinelibrary.wiley.com/doi/pdf/10.1155/2015/258619
    """

    def __init__(self, input_channels, n_classes, kernel_size=None, pool_size=None):

        super(CNN1D, self).__init__()

        if kernel_size is None:

            kernel_size = math.ceil(input_channels / 9)

        if pool_size is None:

            pool_size = math.ceil(kernel_size / 5)

        self.input_channels = input_channels

        self.conv = nn.Conv1d(1, 20, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.features_size = self._get_final_flattened_size()

        self.fc1 = nn.Linear(self.features_size, 100)
        self.fc2 = nn.Linear(100, n_classes)
        self.apply(self.weight_init)

    @staticmethod
    def weight_init(m):
        """
        All trainable parameters of the CNN should be initialized to be
        random values between -0.05 and 0.05
        """
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            nn.init.uniform_(m.weight, -0.05, 0.05)
            nn.init.zeros_(m.bias)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels)
            x = self.pool(self.conv(x))
        return x.numel()

    def forward(self, x):
        """
        Forward pass
        """
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = torch.tanh(self.pool(x))
        x = x.view(-1, self.features_size)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x