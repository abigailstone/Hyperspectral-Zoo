from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms


class HyperspectralDataModule(LightningDataModule):
    """
    `LightningDataModule` for hyperspectral datasets
    """

    def __init__(
        self,
        dataloader,
        n_bands = 220,
        n_classes = 17,
        batch_size = 32,
        val_split = 0.2,
        test_split = 0.1,
        transform=None,
        num_workers = 0,
        pin_memory = False,
    ):
        """
        Initialize a HyperspectralDataModule

        :param dataloader: The dataloader specified in the config
        :param n_bands: The number of bands in this dataset
        :param n_classes: the number of classes in this dataset
        :param batch_size: The batch size. Defaults to 32.
        :param val_split: validation split ratio (0.1 = 10%)
        :param test_split: test split ratio
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.dataloader = dataloader

        self.batch_size_per_device = batch_size

        self.data_train = None
        self.data_val = None
        self.data_test = None


    def setup(self, stage=None) -> None:
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # get dataset using specified dataloader
        dataset = self.dataloader

        # get size of splits
        val_size = int(len(dataset) * self.hparams.val_split)
        test_size = int(len(dataset) * self.hparams.test_split)
        train_size = len(dataset) - val_size - test_size

        # perform train/val/test split
        self.data_train, self.data_val, self.data_test = random_split(
            dataset=dataset,
            lengths=[train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """
        Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True, 
            persistent_workers=True
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """
        Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False, 
            persistent_workers=True
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """
        Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False, 
            persistent_workers=True
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """
        Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = HyperspectralDataModule()