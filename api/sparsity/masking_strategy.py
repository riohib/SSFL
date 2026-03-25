import copy
import random
import concurrent.futures
import os
import pickle
from abc import ABC, abstractmethod
from collections import OrderedDict
from tqdm import tqdm

import torch
import numpy as np
from api.sparsity.saliency_utils import (create_mask_from_scores,
                                         get_mean_saliency_scores)

class BaseMaskingStrategy(ABC):
    """Abstract base class for all mask generation and evolution strategies."""
    def __init__(self, args, logger, device, stats_tracker):
        self.args = args
        self.logger = logger
        self.device = device
        self.tracker = stats_tracker
        self.global_masks = None

    @abstractmethod
    def update_masks_if_needed(
            self, round_idx: int, 
            global_model_w: dict,
            clients: list, 
            prunable_layers: list, 
            max_workers: int,
        ) -> list:
        """
        The main entry point called by the runner each round.
        Returns:
            A list of local_weights_for_agg (empty list if no training occurred).
        """
        pass

    def _create_all_ones_mask(self, global_model_w: dict, prunable_layers: list):
        """
        Creates a mask with all 1's (no pruning) for all prunable layers.
        This is used during warm-up periods.
        """
        self.logger.info("Creating all-1's mask (no pruning during warm-up)...")
        self.global_masks = {}
        for layer_name in prunable_layers:
            if layer_name in global_model_w:
                # Create a mask of all 1's with the same shape as the weight
                self.global_masks[layer_name] = torch.ones_like(
                    global_model_w[layer_name], 
                    device=self.device
                )
        self.logger.info(f"Created all-1's mask for {len(self.global_masks)} layers.")

    def _generate_mask_from_scores(self, global_model_w: dict, clients: list, max_workers: int):
        """
        Generate a new mask based on saliency scores from all clients.
        
        Args:
            global_model_w: Global model weights
            clients: List of clients
            max_workers: Maximum workers for parallel processing
        """
        self.logger.info("Generating global mask based on saliency scores...")
        old_mask = copy.deepcopy(self.global_masks)

        # Calculate saliency scores from all clients in parallel
        all_scores = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_client = {
                executor.submit(self._calculate_scores_for_single_client, client, global_model_w): client
                for client in clients
            }

            pbar = tqdm(
                concurrent.futures.as_completed(future_to_client),
                total=len(clients),
                desc="Calculating Saliency Scores"
            )
            for future in pbar:
                all_scores.append(future.result())

        # Average saliency scores across all clients
        avg_saliency_scores = get_mean_saliency_scores(all_scores)
        
        # Create sparse mask based on averaged saliency scores
        self.global_masks, _ = create_mask_from_scores(
            scores_dict=avg_saliency_scores,
            keep_ratio=self.args.model.dense_ratio,
            device=self.device
        )
        
        # Calculate and log mask change metrics (if we had a previous mask)
        if old_mask is not None:
            self.tracker.calculate_and_store_flip_rate(old_mask, self.global_masks)
            jaccard_dist = self.tracker.calculate_jaccard_distance(old_mask, self.global_masks)
            if jaccard_dist is not None:
                self.tracker.mask_jaccard_distance = jaccard_dist
                self.logger.info(f"Mask Jaccard Distance: {jaccard_dist:.4f}")


    def _calculate_scores_for_single_client(self, client, global_model_w):
        client.strategy.model_trainer.set_model_params(global_model_w)
        return client.generate_saliency_scores(
            method=self.args.algorithm.params.method,
            iterations=self.args.model.itersnip_iteration
        )


class StaticMaskingStrategy(BaseMaskingStrategy):
    """
    Generates a mask only once at the beginning of training (or after warm-up).
    
    Supports optional warm-up period where training occurs with dense (all-1's) masks
    before generating the sparse mask based on saliency scores.
    """
    def update_masks_if_needed(self, round_idx, global_model_w, clients, prunable_layers, max_workers):
        """
        Update masks if needed for static strategy.
        
        Args:
            round_idx: Current training round
            global_model_w: Global model weights
            clients: List of all clients
            prunable_layers: List of prunable parameter names
            max_workers: Maximum number of workers for parallel processing
        """
        warmup_rounds = getattr(self.args.algorithm.params, 'warmup_rounds', 0)
        
        if warmup_rounds > 0:
            # Warm-up mode: use all-1's mask before warmup_rounds
            if round_idx < warmup_rounds:
                if self.global_masks is None:
                    # Initialize all-1's mask on first call
                    self._create_all_ones_mask(global_model_w, prunable_layers)
                # Otherwise, keep existing all-1's mask
            elif round_idx == warmup_rounds:
                # Generate actual sparse mask after warm-up
                self.logger.info(f"Warm-up complete at round {warmup_rounds}. Generating sparse mask from saliency scores...")
                self._generate_mask_from_scores(global_model_w, clients, max_workers)
        else:
            # No warm-up: generate mask at initial_mask_freq (typically round 0)
            if round_idx == self.args.algorithm.params.initial_mask_freq:
                self._generate_mask_from_scores(global_model_w, clients, max_workers)
        
        return []