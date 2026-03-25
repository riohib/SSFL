import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import concurrent.futures
import os

# -----------------------------------------------------------------------------
# 1. Dataset Container and Truncated Dataset Class (No Changes Here)
# -----------------------------------------------------------------------------

class DatasetContainer:
    """
    A simple container class to hold and organize the partitioned dataset.
    """
    def __init__(self):
        self.train_data_local_num_dict = {}
        self.train_data_local_dict = {}
        self.test_data_local_dict = {}
        self.class_num = 10
        self.train_data_num = 0
        self.test_data_num = 0
        self.probabilities = None
        self.traindata_cls_counts = None


class CIFAR10_truncated(data.Dataset):
    """
    A truncated version of the CIFAR10 dataset that works with a subset of indices.
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
        cifar_dataobj = CIFAR10(self.root, self.train, transform=self.transform, download=self.download)
        
        # In torchvision 0.15+, data is named .data, targets is .targets
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

# -----------------------------------------------------------------------------
# 2. Core Partitioning and Loading Logic
# -----------------------------------------------------------------------------

def _get_cifar10_transforms():
    """Returns standard CIFAR-10 transforms."""
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, test_transform

def _partition_cifar10_dirichlet(y_train, n_clients, alpha):
    """
    Partitions data using a Dirichlet distribution, ensuring each client receives
    an equal number of samples. This is a robust and direct implementation of the
    original data loader's logic, guaranteed to terminate.
    """
    n_classes = len(np.unique(y_train))
    n_samples = len(y_train)

    # 1. Define the number of samples per client (ensuring it's balanced)
    samples_per_client = n_samples // n_clients
    client_sample_counts = np.full(n_clients, samples_per_client)
    client_sample_counts[:n_samples % n_clients] += 1

    # 2. Get class priors for each client and prepare data pools
    client_class_priors = np.random.dirichlet(alpha=np.repeat(alpha, n_classes), size=n_clients)
    class_pools = [list(np.where(y_train == i)[0]) for i in range(n_classes)]
    for p in class_pools:
        np.random.shuffle(p)

    # 3. Perform the assignment
    # We create a list of all client "slots" to fill and shuffle it to ensure fairness
    client_indices_map = {i: [] for i in range(n_clients)}
    client_slots = np.repeat(np.arange(n_clients), client_sample_counts)
    np.random.shuffle(client_slots)

    # This loop runs exactly n_samples times, once for each slot. It is guaranteed to finish.
    for client_idx in tqdm(client_slots, desc="Partitioning Training Data"):
        # For the current client slot, determine which class to draw a sample from.
        priors = client_class_priors[client_idx]
        
        # Select ONLY from classes that still have samples available.
        available_classes = [k for k, pool in enumerate(class_pools) if len(pool) > 0]
        
        if not available_classes:
            # This fallback should ideally not be hit if data is distributed correctly
            continue

        # Normalize the client's priors over the available classes
        probs = priors[available_classes]
        normalized_probs = probs / np.sum(probs)
        
        # Choose a class based on the normalized probabilities
        chosen_class = np.random.choice(available_classes, p=normalized_probs)
        
        # Pop the last available sample from that class and assign it
        sample_idx = class_pools[chosen_class].pop()
        client_indices_map[client_idx].append(sample_idx)
            
    # Create final statistics for verification
    final_class_counts = np.zeros((n_clients, n_classes), dtype=int)
    for client_id, indices in client_indices_map.items():
        client_labels = y_train[np.array(indices, dtype=int)]
        final_class_counts[client_id, :] = np.bincount(client_labels, minlength=n_classes)

    return client_indices_map, final_class_counts


# It must be at the top level of the file for the multiprocessing to work.
def _create_client_dataloader(args_tuple):
    """Creates the train/test dataloaders for a single client."""
    client_idx, train_idxs, test_idxs, data_dir, train_transform, test_transform, batch_size = args_tuple

    train_ds_local = CIFAR10_truncated(root=data_dir, dataidxs=train_idxs, train=True, transform=train_transform)
    test_ds_local = CIFAR10_truncated(root=data_dir, dataidxs=test_idxs, train=False, transform=test_transform)

    train_dl_local = DataLoader(
        dataset=train_ds_local, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=False,
    )
    test_dl_local = DataLoader(
        dataset=test_ds_local, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False,
    )
    return client_idx, train_dl_local, test_dl_local, len(train_idxs), len(test_idxs)


# --- YOUR UPDATED FUNCTION (which is correct) ---
def load_and_partition_cifar10(data_dir, partition_method, n_clients, partition_alpha, batch_size, logger, max_workers=None):
    """
    Loads CIFAR-10 data and partitions it among clients using the specified method.
    
    Args:
        max_workers: Maximum number of parallel workers for data loading. 
                     None = use os.cpu_count() (default), set to 4 for RTX systems.
    """
    logger.info("--------- Loading and Partitioning CIFAR-10 Data (New Loader) ---------")

    train_transform, test_transform = _get_cifar10_transforms()
    
    train_dataset = CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(data_dir, train=False, download=True, transform=test_transform)
    
    y_train = np.array(train_dataset.targets)
    y_test = np.array(test_dataset.targets)

    if partition_method.lower() == 'dir':
        logger.info(f"Partitioning training data via Dirichlet (alpha={partition_alpha})")
        net_dataidx_map, traindata_cls_counts = _partition_cifar10_dirichlet(y_train, n_clients, partition_alpha)
    else:
        raise NotImplementedError(f"Partition method '{partition_method}' is not implemented.")

    logger.info("--- Partitioning test data to match training distribution ---")
    test_dataidxs = {i: [] for i in range(n_clients)}
    for k in range(len(np.unique(y_test))):
        idx_k_test = np.where(y_test == k)[0]
        class_k_train_counts = traindata_cls_counts[:, k]
        total_class_k_counts = np.sum(class_k_train_counts)
        if total_class_k_counts > 0:
            proportions = class_k_train_counts / total_class_k_counts
        else: 
            proportions = np.ones(n_clients) / n_clients

        proportions = (proportions * len(idx_k_test)).astype(int)
        remainder = len(idx_k_test) - np.sum(proportions)
        proportions[:remainder] += 1
        current_pos = 0
        for i in range(n_clients):
            num_samples = proportions[i]
            test_dataidxs[i].extend(idx_k_test[current_pos:current_pos + num_samples])
            current_pos += num_samples

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
            results = list(tqdm(executor.map(_create_client_dataloader, task_args), total=n_clients, desc="Creating Client DataLoaders"))
    else:
        logger.info("Using default parallel workers for data loading")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(_create_client_dataloader, task_args), total=n_clients, desc="Creating Client DataLoaders"))

    for client_idx, train_dl, test_dl, num_train, num_test in results:
        dataset_container.train_data_local_dict[client_idx] = train_dl
        dataset_container.test_data_local_dict[client_idx] = test_dl
        dataset_container.train_data_local_num_dict[client_idx] = num_train
        logger.debug(f"Client {client_idx}: {num_train} train samples, {num_test} test samples.")

    dataset_container.train_data_num = len(train_dataset)
    dataset_container.test_data_num = len(test_dataset)
    dataset_container.class_num = len(np.unique(y_train))
    logger.info("--------- Data Loading and Partitioning Complete ---------")
    
    return dataset_container