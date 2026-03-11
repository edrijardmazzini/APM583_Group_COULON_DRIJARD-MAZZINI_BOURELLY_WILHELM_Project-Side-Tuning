# tlkit/data/datasets/pcam_datasets.py

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class PCAMDataset(Dataset):
    """
    PatchCamelyon Dataset - images médicales 96x96 RGB
    Labels: 0 = tissu sain, 1 = métastase

    Les données sont chargées entièrement en mémoire numpy dans __init__
    pour éviter l'erreur "h5py objects cannot be pickled" sur Windows
    avec num_workers > 0.
    """
    def __init__(self, x_path, y_path, transform=None):
        self.transform = transform

        # Charge tout en RAM — évite les problèmes de pickle h5py
        with h5py.File(x_path, 'r') as fx:
            x_key = list(fx.keys())[0]
            self.data = fx[x_key][:]          # numpy array (N, 96, 96, 3) uint8

        with h5py.File(y_path, 'r') as fy:
            y_key = list(fy.keys())[0]
            self.labels = fy[y_key][:].squeeze().astype(np.int64)  # (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]               # (96, 96, 3) numpy uint8
        label = int(self.labels[idx].flatten()[0]) if self.labels[idx].ndim > 0 else int(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label


def get_pcam_datasets(data_dir):
    """
    Retourne (train_dataset, val_dataset, test_dataset).

    Structure attendue dans data_dir :
        training_split.h5
        validation_split.h5
        test_split.h5
        camelyonpatch_level_2_split_train_y.h5
        camelyonpatch_level_2_split_valid_y.h5
        camelyonpatch_level_2_split_test_y.h5
    """
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    transform_eval = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_dataset = PCAMDataset(
        x_path=f'{data_dir}/training_split.h5',
        y_path=f'{data_dir}/camelyonpatch_level_2_split_train_y.h5',
        transform=transform_train
    )
    val_dataset = PCAMDataset(
        x_path=f'{data_dir}/validation_split.h5',
        y_path=f'{data_dir}/camelyonpatch_level_2_split_valid_y.h5',
        transform=transform_eval
    )
    test_dataset = PCAMDataset(
        x_path=f'{data_dir}/test_split.h5',
        y_path=f'{data_dir}/camelyonpatch_level_2_split_test_y.h5',
        transform=transform_eval
    )

    return train_dataset, val_dataset, test_dataset