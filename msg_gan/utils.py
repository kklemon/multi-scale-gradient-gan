from itertools import chain
from pathlib import Path

from torch.utils.data import DataLoader

import config
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.utils import make_grid


def get_dataset(dataset_name, dataset_kwargs=None):
    dataset_builder = config.datasets.get(dataset_name)
    if not dataset_builder:
        raise KeyError(f'No dataset with name \'{dataset_name}\'. Available options are {list(config.datasets.keys())}')

    if not dataset_kwargs:
        dataset_kwargs = {}

    return dataset_builder(**dataset_kwargs)


def get_transform(target_size=None):
    transformations = []

    if target_size:
        transformations.append(transforms.Resize(target_size))

    transformations += [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]

    return transforms.Compose(transformations)


def create_multi_scale_image_grid(samples):
    rescaled = list(samples)

    # Rescale images
    for i, images in enumerate(samples[:-1]):
        rescaled[i] = F.interpolate(images,
                                    scale_factor=2 ** (len(samples) - i - 1),
                                    mode='nearest')

    assert all(map(lambda batch: batch.shape[2:] == samples[-1].shape[2:], rescaled))

    images_per_row = tuple(chain.from_iterable((zip(*rescaled))))
    images = torch.stack(images_per_row)

    return make_grid(images, nrow=len(samples), normalize=True)


def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
