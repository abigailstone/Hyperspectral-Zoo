import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np

class PaviaDataset(Dataset):
    """
    Dataset class for the Pavia dataset
    """

    def __init__(
            self,
            data_dir = "data/",
            data_path = "pavia",
            labels_path = "pavia_gt",
            patch_size = 3,
            transform=None
        ):

        # Load the dataset
        self.data = sio.loadmat(data_dir + data_path)[data_path]
        self.label = sio.loadmat(data_dir + labels_path)[labels_path]

        self.data = self.data.astype(np.float32)
        self.label = self.label.astype(np.int64)

        self.patch_size = patch_size
        self.transform = transform

        # get coordinates
        mask = np.ones_like(self.label)
        x_coord, y_coord = np.nonzero(mask)
        p = patch_size // 2

        # get indices of non-zero values that are within a border of size p
        self.indices = np.array([
            (x, y) for x, y in zip(x_coord, y_coord) if x > p and x < self.data.shape[0] - p and y > p and y < self.data.shape[1] - p
        ])

        self.labels = [self.label[x, y] for x, y in self.indices]


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        x, y = self.indices[idx]

        x1, y1 = x-self.patch_size//2, y-self.patch_size//2
        x2, y2 = x1+self.patch_size, y1+self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype="float32")
        label = np.asarray(np.copy(label), dtype="int64")

        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

        # for 1D CNN
        if self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]

        # for 3D CNN
        if self.patch_size > 1:
            data = data.unsqueeze(0)
            label = label[self.patch_size // 2, self.patch_size // 2]

        return data, label