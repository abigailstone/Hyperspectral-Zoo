import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralSpatialCNN(nn.Module):
    """
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565
    """

    def __init__(self, n_bands, n_classes, patch_size=5, dilation=1):

        super(SpectralSpatialCNN, self).__init__()

        self.n_bands = n_bands
        self.n_classes = n_classes
        self.patch_size = patch_size

        dilation = (dilation, 1, 1)

        self.conv_block1 = nn.Sequential(
            nn.Conv3d(1, 20, kernel_size=(3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=0 if patch_size !=3 else 1),
            nn.ReLU(),
            nn.Conv3d(20, 20, kernel_size=(3, 1, 1), stride=(2, 1, 1), dilation=dilation, padding=(1, 0, 0))
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv3d(20, 35, kernel_size=(3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=(1, 0, 0)),
            nn.ReLU(),
            nn.Conv3d(35, 35, kernel_size=(3, 1, 1), stride=(2, 1, 1), dilation=dilation, padding=(1, 0, 0))
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv3d(35, 35, kernel_size=(3, 1, 1), stride=(1, 1, 1), dilation=dilation, padding=(1, 0, 0)),
            nn.ReLU(),
            nn.Conv3d(35, 35, kernel_size=(2, 1, 1), stride=(2, 1, 1), dilation=dilation, padding=(1, 0, 0)),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(p=0.5)

        self.feature_size = self._get_flattened_size()

        self.fc = nn.Linear(self.feature_size, n_classes)

        self.apply(self.weight_init)

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def _get_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_bands, self.patch_size, self.patch_size)
            x = self.conv_block1(x)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            _, t, c, w, h = x.size()
        return t * c* w * h

    def forward(self, x):
        """
        Forward pass of the model
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x_flat = x.view(-1, self.feature_size)
        x_flat = self.dropout(x_flat)
        x = self.fc(x_flat)

        return x
