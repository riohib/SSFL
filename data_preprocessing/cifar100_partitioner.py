"""
CIFAR-100 specific data partitioning and loading utilities.
"""
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm
import concurrent.futures
import os

from .partition_utils import DatasetContainer, partition_data_dirichlet, partition_test_data_proportional


def _get_cifar100_transforms():
    """Returns standard CIFAR-100 transforms."""
    CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
    CIFAR100_STD = [0.2675, 0.2565, 0.2761]
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    return train_transform, test_transform


class CIFAR100_truncated(data.Dataset):
    """
    A truncated version of the CIFAR100 dataset that works with a subset of indices.
    """
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        cifar_dataobj = CIFAR100(self.root, self.train, transform=self.transform, download=self.download)
        
        data = cifar_dataobj.data
        target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        
        return img, target

    def __len__(self):
        return len(self.data)


def _create_client_dataloader_cifar100(args_tuple):
    """Creates the train/test dataloaders for a single client."""
    client_idx, train_idxs, test_idxs, data_dir, train_transform, test_transform, batch_size = args_tuple

    train_ds_local = CIFAR100_truncated(root=data_dir, dataidxs=train_idxs, train=True, transform=train_transform)
    test_ds_local = CIFAR100_truncated(root=data_dir, dataidxs=test_idxs, train=False, transform=test_transform)

    train_dl_local = DataLoader(
        dataset=train_ds_local, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=False,
        num_workers=0,           # Change from 2 to 0
        pin_memory=False,        # Change from True to False
        persistent_workers=False # Change from True to False
    )
    test_dl_local = DataLoader(
        dataset=test_ds_local, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False,
        num_workers=0,           # Change from 2 to 0
        pin_memory=False,        # Change from True to False
        persistent_workers=False # Change from True to False
    )
    return client_idx, train_dl_local, test_dl_local, len(train_idxs), len(test_idxs)


def load_and_partition_cifar100(data_dir, partition_method, n_clients, partition_alpha, batch_size, logger, max_workers=None):
    """
    Loads CIFAR-100 data and partitions it among clients using the specified method.
    
    Args:
        max_workers: Maximum number of parallel workers for data loading. 
                     None = use os.cpu_count() (default), set to 4 for RTX systems.
    """
    logger.info("--------- Loading and Partitioning CIFAR-100 Data ---------")

    train_transform, test_transform = _get_cifar100_transforms()
    
    train_dataset = CIFAR100(data_dir, train=True, download=True, transform=train_transform)
    test_dataset = CIFAR100(data_dir, train=False, download=True, transform=test_transform)
    
    y_train = np.array(train_dataset.targets)
    y_test = np.array(test_dataset.targets)

    if partition_method.lower() == 'dir':
        logger.info(f"Partitioning training data via Dirichlet (alpha={partition_alpha})")
        net_dataidx_map, traindata_cls_counts = partition_data_dirichlet(y_train, n_clients, partition_alpha)
    else:
        raise NotImplementedError(f"Partition method '{partition_method}' is not implemented.")

    logger.info("--- Partitioning test data to match training distribution ---")
    test_dataidxs = partition_test_data_proportional(y_test, traindata_cls_counts, n_clients)

    dataset_container = DatasetContainer()
    dataset_container.traindata_cls_counts = traindata_cls_counts

    logger.info("--- Creating Client DataLoaders in parallel ---")
    
    task_args = [
        (i, net_dataidx_map[i], test_dataidxs[i], data_dir, train_transform, test_transform, batch_size) 
        for i in range(n_clients)
    ]
    
    # Only pass max_workers if explicitly set in config, otherwise use ProcessPoolExecutor default
    if max_workers is not None:
        logger.info(f"Using {max_workers} parallel workers for data loading")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(_create_client_dataloader_cifar100, task_args), total=n_clients, desc="Creating Client DataLoaders"))
    else:
        logger.info("Using default parallel workers for data loading")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(_create_client_dataloader_cifar100, task_args), total=n_clients, desc="Creating Client DataLoaders"))

    for client_idx, train_dl, test_dl, num_train, num_test in results:
        dataset_container.train_data_local_dict[client_idx] = train_dl
        dataset_container.test_data_local_dict[client_idx] = test_dl
        dataset_container.train_data_local_num_dict[client_idx] = num_train
        logger.debug(f"Client {client_idx}: {num_train} train samples, {num_test} test samples.")

    dataset_container.train_data_num = len(train_dataset)
    dataset_container.test_data_num = len(test_dataset)
    dataset_container.class_num = len(np.unique(y_train))  # 100 for CIFAR-100
    logger.info("--------- CIFAR-100 Data Loading and Partitioning Complete ---------")
    
    return dataset_container