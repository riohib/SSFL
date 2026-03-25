"""
Utility functions for data partitioning that can be shared across different datasets.
"""
import numpy as np
from tqdm import tqdm
import copy


class DatasetContainer:
    """
    A simple container class to hold and organize the partitioned dataset.
    """
    def __init__(self):
        self.train_data_local_num_dict = {}
        self.train_data_local_dict = {}
        self.test_data_local_dict = {}
        self.class_num = 10  # Will be updated based on dataset
        self.train_data_num = 0
        self.test_data_num = 0
        self.probabilities = None
        self.traindata_cls_counts = None


def partition_data_dirichlet(y_train, n_clients, alpha):
    """
    Partitions data using a Dirichlet distribution, ensuring each client receives
    an equal number of samples. This is a robust and direct implementation that
    works for any number of classes.
    
    Args:
        y_train: Array of training labels
        n_clients: Number of clients to partition data among
        alpha: Dirichlet concentration parameter (lower = more heterogeneous)
        
    Returns:
        tuple: (client_indices_map, traindata_cls_counts)
            - client_indices_map: Dict mapping client_id -> list of sample indices
            - traindata_cls_counts: Array of shape (n_clients, n_classes) with class counts
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
    client_indices_map = {i: [] for i in range(n_clients)}
    client_slots = np.repeat(np.arange(n_clients), client_sample_counts)
    np.random.shuffle(client_slots)

    # This loop runs exactly n_samples times, once for each slot
    for client_idx in tqdm(client_slots, desc="Partitioning Training Data"):
        priors = client_class_priors[client_idx]
        
        # Select ONLY from classes that still have samples available
        available_classes = [k for k, pool in enumerate(class_pools) if len(pool) > 0]
        
        if not available_classes:
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


def partition_test_data_proportional(y_test, traindata_cls_counts, n_clients):
    """
    Partitions test data to match the training distribution proportionally.
    
    Args:
        y_test: Array of test labels
        traindata_cls_counts: Array of shape (n_clients, n_classes) from training partition
        n_clients: Number of clients
        
    Returns:
        dict: Mapping client_id -> list of test sample indices
    """
    test_dataidxs = {i: [] for i in range(n_clients)}
    n_classes = len(np.unique(y_test))
    
    for k in range(n_classes):
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
            
    return test_dataidxs


def partition_data_pathological(y_train, n_clients, classes_per_client):
    """
    Partitions data using pathological (non-IID) distribution where each client
    gets data from exactly `classes_per_client` classes.
    
    This creates extreme non-IID distribution - each client is a "specialist" 
    for a small subset of classes.
    
    Args:
        y_train: Array of training labels
        n_clients: Number of clients to partition data among
        classes_per_client: Number of classes each client should have (e.g., 2)
        
    Returns:
        tuple: (client_indices_map, traindata_cls_counts)
            - client_indices_map: Dict mapping client_id -> list of sample indices
            - traindata_cls_counts: Array of shape (n_clients, n_classes) with class counts
    """
    import random
    
    n_classes = len(np.unique(y_train))
    n_samples = len(y_train)
    
    # Verify we have enough class combinations
    if classes_per_client > n_classes:
        raise ValueError(f"classes_per_client ({classes_per_client}) cannot exceed n_classes ({n_classes})")
    
    # Create class pools
    class_pools = [list(np.where(y_train == i)[0]) for i in range(n_classes)]
    for p in class_pools:
        random.shuffle(p)
    
    # Assign classes to clients
    # Generate all possible class combinations
    from itertools import combinations
    all_combinations = list(combinations(range(n_classes), classes_per_client))
    
    # If we have more clients than combinations, we'll reuse combinations
    # If we have fewer clients, randomly sample combinations
    if n_clients <= len(all_combinations):
        selected_combinations = random.sample(all_combinations, n_clients)
    else:
        # Repeat combinations to cover all clients
        selected_combinations = (all_combinations * ((n_clients // len(all_combinations)) + 1))[:n_clients]
        random.shuffle(selected_combinations)
    
    # Initialize client indices map
    client_indices_map = {i: [] for i in range(n_clients)}
    
    # Calculate samples per client (balanced)
    samples_per_client = n_samples // n_clients
    remainder = n_samples % n_clients
    
    # Assign samples to clients based on their assigned classes
    for client_idx in range(n_clients):
        assigned_classes = selected_combinations[client_idx]
        
        # Calculate how many samples this client should get
        client_sample_count = samples_per_client + (1 if client_idx < remainder else 0)
        
        # Distribute samples evenly among assigned classes
        samples_per_class = client_sample_count // len(assigned_classes)
        extra_samples = client_sample_count % len(assigned_classes)
        
        for i, class_idx in enumerate(assigned_classes):
            # Number of samples for this class
            num_samples = samples_per_class + (1 if i < extra_samples else 0)
            
            # Take samples from the class pool
            for _ in range(num_samples):
                if len(class_pools[class_idx]) > 0:
                    sample_idx = class_pools[class_idx].pop()
                    client_indices_map[client_idx].append(sample_idx)
    
    # Create final statistics for verification
    final_class_counts = np.zeros((n_clients, n_classes), dtype=int)
    for client_id, indices in client_indices_map.items():
        if len(indices) > 0:
            client_labels = y_train[np.array(indices, dtype=int)]
            final_class_counts[client_id, :] = np.bincount(client_labels, minlength=n_classes)
    
    return client_indices_map, final_class_counts