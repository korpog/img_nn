import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

training_data = datasets.GTSRB(
    root="data",
    split="train",
    download=True,
    transform=ToTensor()
)

test_data = datasets.GTSRB(
    root="data",
    split="test",
    download=True,
    transform=ToTensor()
)