from pathlib import Path

import PIL.Image
from torch.utils import data


class FlatImageFolder(data.Dataset):
    def __init__(self, root, transform=None, ext=('.png', '.jpg', '.jpeg', 'bmp')):
        super().__init__()

        self.files = [f for f in Path(root).iterdir() if f.is_file() and f.suffix in ext]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        X = PIL.Image.open(self.files[idx])
        if self.transform:
            X = self.transform(X)
        return X, []


class IgnoreLabelDataset(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx][0]

    def __len__(self):
        return len(self.dataset)


class TransformDataset(data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)
