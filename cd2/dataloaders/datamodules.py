import logging
import os
from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from cd2.utils.fdataclasses import collate_batch

class DiffusionDataset(Dataset):
    def __init__(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        standardize: bool = False,
        X_ref: Optional[torch.Tensor] = None,
    ):
        """Dataset for diffusion models.

        Args:
            X (torch.Tensor): Time series that are fed to the model.
            y (Optional[torch.Tensor], optional): Potential labels. Defaults to None.
            standardize (bool, optional): Standardize each feature in the dataset. Defaults to False.
            X_ref (Optional[torch.Tensor], optional): Features used to compute the mean and std. Defaults to None.
        """

        super().__init__()
        self.X = X
        self.y = y
        self.standardize = standardize
        if X_ref is None:
            self.X_ref = X
        else:
            self.X_ref = X_ref

        assert isinstance(X_ref, torch.Tensor)
        self.X_mean = torch.mean(self.X_ref, dim=0)
        self.X_std = torch.std(self.X_ref, dim=0)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = {}
        data["X"] = self.X[idx]
        if self.standardize:
            data["X"] = (data["X"] - self.X_mean) / self.X_std
        if self.y is not None:
            data["y"] = self.y[idx]
        return data
    
class DataModule(pl.LightningDataModule, ABC):
    def __init__(
        self,
        data_dir : Path | str = Path.cwd() / "data",
        random_seed: int = 42,
        batch_size: int = 32,
        standardize: bool = False,
    ):
        super().__init__()
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        self.data_dir = data_dir / self.dataset_name
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.standardize = standardize
        self.X_train = torch.Tensor()
        self.y_train: Optional[torch.Tensor] = None
        self.X_test = torch.Tensor()
        self.y_test: Optional[torch.Tensor] = None

    def train_dataloader(self) -> DataLoader:
        train_set = DiffusionDataset(
            X = self.X_train,
            y = self.y_train,
            standardize = self.standardize
        )
        return DataLoader(
            train_set,
            batch_size = self.batch_size,
            shuffle = True,
            collate_fn = collate_batch
        )
    
    def test_dataloader(self) -> DataLoader:
        test_dataset = DiffusionDataset(
            X = self.X_test,
            y = self.y_test,
        )
        return DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_batch
        )
    
    def val_dataloader(self) -> DataLoader:
        test_set = DiffusionDataset(
            X = self.X_test,
            y = self.y_test,
            standardize = self.standardize,
            X_ref = self.X_train
        )
        return DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_batch
        )

    @abstractproperty
    def dataset_name(self) -> str: ...

    @property
    def dataset_parameters(self) -> dict[str, Any]:
        return {
            "in_channels": self.X_train.size(2),
            "seq_len": self.X_train.size(1),
            "num_training_steps": len(self.train_dataloader())
        }
    
    @property
    def feature_mean_and_std(self) -> tuple[torch.Tensor, torch.Tensor]:
        trainset = DiffusionDataset(
            X = self.X_train,
            y = self.y_train,
            standardize = self.standardize
        )

        return trainset.X_mean, trainset.X_std