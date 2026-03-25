# In a new file: api/runners.py

import copy
from abc import ABC, abstractmethod
from collections import OrderedDict
import random
import concurrent.futures

import numpy as np
import torch
import wandb

from conf.config_schema import Config
from api.client.client import Client
from api.utils.stats_tracker import StatsTracker


class BaseRunner(ABC):
    """
    An abstract base class for all federated learning algorithm runners.
    It defines a common interface for initializing, setting up clients, and training.
    """
    def __init__(self, dataset, device, args: Config, model_trainer, logger):
        self.dataset = dataset
        self.device = device
        self.args = args
        self.logger = logger
        self.model_trainer = model_trainer
        self.client_list = []

        self.tracker = StatsTracker(self.args.wandb.mode, self.logger)
        
        # Unpack necessary dataset attributes
        self.train_data_local_num_dict = self.dataset.train_data_local_num_dict
        self.train_data_local_dict = self.dataset.train_data_local_dict
        self.test_data_local_dict = self.dataset.test_data_local_dict

    @property
    def _max_workers(self):
        """Simple helper to get max workers from config."""
        if self.args.training.max_workers > 0:
            return self.args.training.max_workers
        return self.args.training.client_num_in_total
    
    @abstractmethod
    def train(self):
        """
        The main entry point to start the algorithm's training process.
        This method must be implemented by all subclasses.
        """
        pass

    def _setup_clients(self, strategy_class):
        """
        A concrete method to set up clients using a provided strategy class.
        This method is now DRY and shared by all runners.
        """
        self.logger.info(f"Setting up clients with {strategy_class.__name__}...")
        for client_idx in range(self.args.training.client_num_in_total):
            client = Client(
                client_idx=client_idx,
                local_training_data=self.train_data_local_dict.get(client_idx),
                local_test_data=self.test_data_local_dict.get(client_idx),
                local_sample_number=self.train_data_local_num_dict.get(client_idx),
                args=self.args,
                device=self.device,
                model_trainer=copy.deepcopy(self.model_trainer),
                logger=self.logger,
                strategy_class=strategy_class  # Use the passed-in strategy
            )
            self.client_list.append(client)
        self.logger.info(f"Successfully set up {len(self.client_list)} clients.")