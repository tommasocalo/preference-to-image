from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import torch

class DummyDataset(Dataset):
    """A simple dataset that contains a fixed number of dummy items."""
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return a dummy tensor; actual data is not important
        return torch.tensor([0])

class BaselineDataModule(LightningDataModule):
    """LightningDataModule for managing dummy data loaders for Bayesian Optimization."""
    def __init__(self, num_iterations: int = 1000):
        super().__init__()
        self.num_iterations = num_iterations

    def setup(self, stage=None):
        """Setup the dataset (called automatically by the trainer)."""
        if stage == 'fit' or stage is None:
            self.train_dataset = DummyDataset(self.num_iterations)

    def train_dataloader(self):
        """Create a DataLoader to control the number of iterations in the training loop."""
        return DataLoader(self.train_dataset, batch_size=1)
