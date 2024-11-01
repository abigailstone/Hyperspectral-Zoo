import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleCNN(nn.Module):
    """
    https://ieeexplore.ieee.org/document/8297014/
    """

    def __init__(self, n_bands, n_classes, patch_size) -> None:

        super(MultiScaleCNN, self).__init__()

        self.n_bands = n_bands
        self.n_classes = n_classes
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, 16, (11, 3, 3), stride=(3, 1, 1))

        self.conv2_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0))
        self.conv2_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1, 0, 0))
        self.conv2_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2, 0, 0))
        self.conv2_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5, 0, 0))

        self.conv3_1 = nn.Conv3d(16, 16, (1, 1, 1), padding=(0, 0, 0))
        self.conv3_2 = nn.Conv3d(16, 16, (3, 1, 1), padding=(1, 0, 0))
        self.conv3_3 = nn.Conv3d(16, 16, (5, 1, 1), padding=(2, 0, 0))
        self.conv3_4 = nn.Conv3d(16, 16, (11, 1, 1), padding=(5, 0, 0))

        self.conv4 = nn.Conv3d(16, 16, (3, 2, 2))

        self.pooling = nn.MaxPool2d((3, 2, 2), stride=(3, 2, 2))
        self.dropout = nn.Dropout(p=0.6)

        self.features_size = self._get_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)
        self.apply(self.weight_init)

    @staticmethod
    def weight_init(m):
        """
        Weight initialization
        """
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def _get_flattened_size(self):
        """
        Get flattened size of features
        """
        x = torch.zeros((1, 1, self.n_bands, self.patch_size, self.patch_size))

        x = self.conv1(x)
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x2_3 = self.conv2_3(x)
        x2_4 = self.conv2_4(x)
        x = x2_1 + x2_2 + x2_3 + x2_4
        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(x)
        x3_3 = self.conv3_3(x)
        x3_4 = self.conv3_4(x)
        x = x3_1 + x3_2 + x3_3 + x3_4
        x = self.conv4(x)

        _, t, c, w, h = x.size()

        return t * c * w * h

    def forward(self, x):
        """
        Forward pass
        """
        x = self.conv1(x)
        x = F.relu(x)

        # Multi-scale block 1
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x2_3 = self.conv2_3(x)
        x2_4 = self.conv2_4(x)

        x = x2_1 + x2_2 + x2_3 + x2_4
        x = F.relu(x)

        # Multi-scale block 2
        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(x)
        x3_3 = self.conv3_3(x)
        x3_4 = self.conv3_4(x)

        x = x3_1 + x3_2 + x3_3 + x3_4
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = x.view(-1, self.features_size)
        x = self.dropout(x)
        x = self.fc(x)

        return x


