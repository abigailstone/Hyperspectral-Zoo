import pytest
import torch

from src.data.hyperspectral_datamodule import HyperspectralDataModule
from src.data.components.indian_pines import IndianPinesDataset


@pytest.mark.parametrize("batch_size", [32, 128])
def test_hyperspectral_datamodule(batch_size: int) -> None:
    """
    Tests for HyperspectralDatamodule
    """
    data_dir = "data/"

    dataloader = IndianPinesDataset(data_dir)

    dm = HyperspectralDataModule(dataloader=dataloader, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64

    batch = next(iter(dm.val_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
