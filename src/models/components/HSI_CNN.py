import torch
import torch.nn as nn
import torch.nn.functional as F

class HSICNN(nn.Module):
    """
    https://arxiv.org/pdf/1802.10478
    """

    def __init__(self, n_bands, n_classes, patch_size, n_planes):

        super(HSICNN, self).__init__()

        self.n_bands = n_bands 
        self.n_classes = n_classes 
        self.patch_size = patch_size 
        self.n_planes = n_planes

        self.conv1 = nn.Conv3d(1, 90, (24, 3, 3), padding=0, stride=(9, 1, 1))
        self.conv2 = nn.Conv2d(1, 64, (3, 3), padding=0, stride=(1, 1))

        self.features_size = self._get_flattened_size()

        self.fc1 = nn.Linear(self.features_size, 1024)
        self.fc2 = nn.Linear(1024, self.n_classes)

    @staticmethod
    def weight_init(m):
        """
        Weight initialization
        """
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def _get_flattened_size(self):
        """
        Get flattened size of features
        """

        with torch.no_grad():

            x = torch.zeros(
                (1, 1, self.n_bands, self.patch_size, self.patch_size)
            )

            x = self.conv1(x)
            x = x.view(x.size(0), 1, -1, self.n_planes)
            x = self.conv2(x)
            _, c, w, h = x.size()

        return c * w * h

    def forward(self, x):
        """
        Forward pass
        """

        # 3D conv layers 
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), 1, -1, self.n_planes)
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.features_size)

        # fully-connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

