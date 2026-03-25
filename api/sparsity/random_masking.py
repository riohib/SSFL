import copy
import random
import concurrent.futures
from abc import ABC, abstractmethod
from collections import OrderedDict
from tqdm import tqdm

import torch
from api.sparsity.saliency_utils import (create_mask_from_scores, get_mean_saliency_scores)
from api.sparsity.masking_strategy import BaseMaskingStrategy

class GlobalRandomMaskingStrategy(BaseMaskingStrategy):
    """
    Generates a single global random mask once at the beginning of training.
    The dense_ratio is applied across all prunable layers collectively.
    """
    def update_masks_if_needed(self, round_idx, global_model_w, clients, prunable_layers, max_workers,
                               ood_experiment_mode=None, ood_introduction_round=None, ood_client_ids=None):
        """
        Calls the random mask generation method at the designated round.
        """
        warmup_rounds = getattr(self.args.algorithm.params, 'warmup_rounds', 0)
        
        if warmup_rounds > 0:
            # Warm-up mode: use all-1's mask before round warmup_rounds
            if round_idx < warmup_rounds:
                if self.global_masks is None:
                    # Initialize all-1's mask on first call
                    self._create_all_ones_mask(global_model_w, prunable_layers)
                # Otherwise, keep existing all-1's mask
            elif round_idx == warmup_rounds:
                # Generate actual random mask after warm-up
                self.logger.info(f"Warm-up complete at round {warmup_rounds}. Generating global random mask...")
                self._generate_global_random_mask(global_model_w, prunable_layers)
        else:
            # Original behavior: generate mask at initial_mask_freq
            if round_idx == self.args.algorithm.params.initial_mask_freq:
                self._generate_global_random_mask(global_model_w, prunable_layers)
        return [] # No training is performed by this strategy

    def _generate_global_random_mask(self, global_model_w: dict, prunable_layers: list):
        """
        Creates one large mask for the whole model and distributes it back to the layers,
        ensuring the overall model density matches the target.
        """
        self.logger.info(f"Generating a single GLOBAL RANDOM mask with dense_ratio = {self.args.model.dense_ratio}...")
        
        all_weights_flat = torch.cat([
            global_model_w[name].flatten() for name in prunable_layers if name in global_model_w
        ])
        
        if len(all_weights_flat) == 0:
            self.logger.error("Could not find any prunable layers. Cannot generate global random mask.")
            return

        total_params = len(all_weights_flat)
        num_params_to_keep = int(total_params * self.args.model.dense_ratio)
        
        # Create a single flat mask for the entire model
        flat_mask = torch.zeros_like(all_weights_flat)
        indices_to_keep = torch.randperm(total_params)[:num_params_to_keep]
        flat_mask[indices_to_keep] = 1.0

        # Distribute the flat mask back to the individual layers
        self.global_masks = {}
        current_pos = 0
        for name in prunable_layers:
            if name in global_model_w:
                layer_shape = global_model_w[name].shape
                layer_numel = global_model_w[name].numel()
                
                layer_mask_flat = flat_mask[current_pos : current_pos + layer_numel]
                self.global_masks[name] = layer_mask_flat.reshape(layer_shape).to(self.device)
                current_pos += layer_numel

        self.logger.info("Successfully generated and stored the GLOBAL random mask.")


class LayerwiseRandomMaskingStrategy(BaseMaskingStrategy):
    """
    Generates a random mask for each prunable layer independently once at the
    beginning of training. The dense_ratio is applied to each layer.
    """
    def update_masks_if_needed(self, round_idx, global_model_w, clients, prunable_layers, max_workers,
                               ood_experiment_mode=None, ood_introduction_round=None, ood_client_ids=None):
        """
        Calls the random mask generation method at the designated round.
        """
        warmup_rounds = getattr(self.args.algorithm.params, 'warmup_rounds', 0)
        
        if warmup_rounds > 0:
            # Warm-up mode: use all-1's mask before round warmup_rounds
            if round_idx < warmup_rounds:
                if self.global_masks is None:
                    # Initialize all-1's mask on first call
                    self._create_all_ones_mask(global_model_w, prunable_layers)
                # Otherwise, keep existing all-1's mask
            elif round_idx == warmup_rounds:
                # Generate actual random mask after warm-up
                self.logger.info(f"Warm-up complete at round {warmup_rounds}. Generating layer-wise random masks...")
                self._generate_layerwise_random_mask(global_model_w, prunable_layers)
        else:
            # Original behavior: generate mask at initial_mask_freq
            if round_idx == self.args.algorithm.params.initial_mask_freq:
                self._generate_layerwise_random_mask(global_model_w, prunable_layers)
        return [] # No training is performed by this strategy

    def _generate_layerwise_random_mask(self, global_model_w: dict, prunable_layers: list):
        """
        Creates a new mask for each prunable layer, ensuring that each layer
        individually meets the target density.
        """
        self.logger.info(f"Generating LAYER-WISE RANDOM masks with dense_ratio = {self.args.model.dense_ratio}...")
        self.global_masks = {}
        
        for name in prunable_layers:
            if name in global_model_w:
                layer_weights = global_model_w[name]
                layer_numel = layer_weights.numel()
                num_to_keep = int(layer_numel * self.args.model.dense_ratio)
                
                # Create a mask for this specific layer
                layer_mask_flat = torch.zeros(layer_numel, device=self.device)
                indices_to_keep = torch.randperm(layer_numel, device=self.device)[:num_to_keep]
                layer_mask_flat[indices_to_keep] = 1.0
                
                # Reshape and store the mask
                self.global_masks[name] = layer_mask_flat.reshape(layer_weights.shape)
        
        self.logger.info("Successfully generated and stored LAYER-WISE random masks.")


class ClientwiseRandomMaskingStrategy(BaseMaskingStrategy):
    """
    Generates a separate random mask for each client independently.
    Each client gets its own random mask at the specified dense_ratio.
    This is a baseline for personalized federated learning.
    """
    def __init__(self, args, logger, device, stats_tracker):
        super().__init__(args, logger, device, stats_tracker)
        self.client_masks = {}  # Dict mapping client_idx -> mask dict
        
    def update_masks_if_needed(self, round_idx, global_model_w, clients, prunable_layers, max_workers,
                               ood_experiment_mode=None, ood_introduction_round=None, ood_client_ids=None):
        """
        Generates random masks for each client at the designated round.
        """
        warmup_rounds = getattr(self.args.algorithm.params, 'warmup_rounds', 0)
        
        if warmup_rounds > 0:
            # Warm-up mode: use all-1's mask before round warmup_rounds
            if round_idx < warmup_rounds:
                if not self.client_masks:
                    # Initialize all-1's masks for all clients
                    self._create_all_ones_masks_for_all_clients(global_model_w, prunable_layers, clients)
            elif round_idx == warmup_rounds:
                # Generate actual random masks after warm-up
                self.logger.info(f"Warm-up complete at round {warmup_rounds}. Generating client-wise random masks...")
                self._generate_clientwise_random_masks(global_model_w, prunable_layers, clients)
        else:
            # Original behavior: generate mask at initial_mask_freq
            if round_idx == self.args.algorithm.params.initial_mask_freq:
                self._generate_clientwise_random_masks(global_model_w, prunable_layers, clients)
        
        # For client-wise masks, we also need to set global_masks to None or a placeholder
        # The runner will use client_masks instead
        self.global_masks = None
        return []  # No training is performed by this strategy
    
    def _create_all_ones_masks_for_all_clients(self, global_model_w: dict, prunable_layers: list, clients: list):
        """Creates all-1's masks for all clients during warm-up."""
        self.logger.info("Creating all-1's masks for all clients (no pruning during warm-up)...")
        for client in clients:
            mask = {}
            for layer_name in prunable_layers:
                if layer_name in global_model_w:
                    mask[layer_name] = torch.ones_like(
                        global_model_w[layer_name], 
                        device=self.device
                    )
            self.client_masks[client.client_idx] = mask
        self.logger.info(f"Created all-1's masks for {len(self.client_masks)} clients.")
    
    def _generate_clientwise_random_masks(self, global_model_w: dict, prunable_layers: list, clients: list):
        """
        Generates a separate random mask for each client.
        Each mask is generated independently with the specified dense_ratio.
        """
        self.logger.info(f"Generating CLIENT-WISE RANDOM masks with dense_ratio = {self.args.model.dense_ratio}...")
        self.client_masks = {}
        
        for client in clients:
            client_mask = {}
            for name in prunable_layers:
                if name in global_model_w:
                    layer_weights = global_model_w[name]
                    layer_numel = layer_weights.numel()
                    num_to_keep = int(layer_numel * self.args.model.dense_ratio)
                    
                    # Create a random mask for this client's layer
                    layer_mask_flat = torch.zeros(layer_numel, device=self.device)
                    indices_to_keep = torch.randperm(layer_numel, device=self.device)[:num_to_keep]
                    layer_mask_flat[indices_to_keep] = 1.0
                    
                    # Reshape and store the mask
                    client_mask[name] = layer_mask_flat.reshape(layer_weights.shape)
            
            self.client_masks[client.client_idx] = client_mask
        
        self.logger.info(f"Successfully generated and stored CLIENT-WISE random masks for {len(self.client_masks)} clients.")
    
    def get_mask_for_client(self, client_idx: int):
        """Returns the mask for a specific client."""
        return self.client_masks.get(client_idx, None)