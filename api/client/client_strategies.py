# In a new file: api/client_strategies.py
from abc import ABC, abstractmethod
from collections import namedtuple
import copy
import math
import torch
import numpy as np

from ..sparsity.saliency import get_saliency_scores
from api.sparsity.sparse_tools import calculate_model_sparsity_from_weights


TrainingResult = namedtuple('TrainingResult', [
    'weights', 'grads', 'training_flops', 'comm_params', 'avg_loss', 'sparsity_dict', 'learning_rate'
])

class BaseClientStrategy(ABC):
    """Abstract base class for all client-side training strategies."""
    def __init__(self, client, args, device, model_trainer, logger):
        self.client = client
        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.logger = logger

    @abstractmethod
    def train(self, w, masks, round):
        """Performs one round of local training."""
        pass


class SSFLClientStrategy(BaseClientStrategy):
    """Contains all client-side logic for SSFL (Sparse Salient Federated Learning) algorithm."""

    def train(self, global_model_w, masks, round_num):
        self.model_trainer.model.load_state_dict(global_model_w)
        self.model_trainer.model.to(self.device)

        if masks:
            self.model_trainer.model.apply_masks(masks)

        self.model_trainer.set_id(self.client.client_idx)
        avg_loss, final_grads, final_lr = self.model_trainer.train(
            self.client.local_training_data, self.device, self.args, round_num, masks
        )
        
        weights = self.model_trainer.get_model_params()
        
        sparsity_dict = calculate_model_sparsity_from_weights(
            weights, self.model_trainer.prunable_parameter_names
        )

        training_flops = self.args.training.epochs * self.client.local_sample_number * self.model_trainer.count_training_flops_per_sample()
        num_comm_params = self.model_trainer.count_communication_params(weights)
        
        return TrainingResult(
            weights=weights, grads=final_grads, training_flops=training_flops,
            comm_params=num_comm_params, avg_loss=avg_loss,
            sparsity_dict=sparsity_dict, learning_rate=final_lr
        )

    def generate_saliency_scores(self, method='ssfl', iterations=1):
        if iterations == 0: return {}

        first_batch = next(iter(self.client.local_training_data))
        total_scores = get_saliency_scores(method, self.model_trainer, first_batch, self.logger)

        for i in range(1, iterations):
            mini_batch = next(iter(self.client.local_training_data))
            scores = get_saliency_scores(method, self.model_trainer, mini_batch, self.logger)
            for k, v in scores.items():
                if k in total_scores: total_scores[k] += v

        for k in total_scores:
            total_scores[k] /= iterations
            
        return total_scores