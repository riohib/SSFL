import copy
import os
import random
from collections import OrderedDict
from typing import List
import concurrent.futures

import numpy as np
import torch
import torch.nn as nn
import wandb

from conf.config_schema import Config
from api.client.client import Client
from api.sparsity.sparse_tools import get_threshold_list, calculate_model_sparsity_from_weights
from api.utils.stats_tracker import StatsTracker
from api.trainers.training_utils import save_final_checkpoint, save_intermediate_checkpoint
from api.algorithms.base_runner import BaseRunner
from api.client.client_strategies import SSFLClientStrategy
from api.sparsity.masking_strategy import StaticMaskingStrategy
from api.sparsity.random_masking import GlobalRandomMaskingStrategy, LayerwiseRandomMaskingStrategy, ClientwiseRandomMaskingStrategy

class SSFL(BaseRunner):
    """
    SSFL (Sparse Salient Federated Learning) - Main federated learning orchestrator.
    
    This class sets up clients and manages the overall training, mask generation,
    and testing process across communication rounds with sparse global masks.
    """

    def __init__(self, dataset, device, args: Config, model_trainer, logger):
        """
        Initializes the SSFL runner by setting up all necessary state.
        """
        super().__init__(dataset, device, args, model_trainer, logger)

        # Create the appropriate masking strategy based on the config
        self.masking_strategy = self._create_masking_strategy()
        
        # The runner's global_masks is a reference to the strategy's mask.
        #   It will be None initially and populated later by the strategy.
        self.global_masks = self.masking_strategy.global_masks 

        # Initialize model weights and aggregation strategy
        self.global_model_w = self.model_trainer.get_model_params()
        self.w_initial = copy.deepcopy(self.global_model_w)
        
        # Set the aggregation method
        self.aggregation_strategies = { "FedAvg": self._aggregate_fedavg, "topk_local": self._aggregate_topk }
        self.averaging_mode = "topk_local" if self.args.algorithm.params.method == "topk_local" else "FedAvg"

        # Set up the clients
        self._setup_clients(strategy_class=SSFLClientStrategy)



    def _create_masking_strategy(self):
        """
        Factory method to create the appropriate masking strategy.
        
        Returns:
            BaseMaskingStrategy: Instance of the configured masking strategy
                (Static, Dynamic, Evolve, or Random variants)
        """
        mode = self.args.algorithm.params.mode
        self.logger.info(f"Initializing SSFL runner with masking strategy: '{mode}'")

        # Pass all common dependencies to the strategy constructors
        common_args = (self.args, self.logger, self.device, self.tracker)

        if mode == 'static':
            return StaticMaskingStrategy(*common_args)
        elif mode == 'random_global':
            self.logger.info("Using Global Random masking strategy.")
            return GlobalRandomMaskingStrategy(*common_args)
        elif mode == 'random_layerwise':
            self.logger.info("Using Layer-wise Random masking strategy.")
            return LayerwiseRandomMaskingStrategy(*common_args)
        elif mode == 'random_clientwise':
            self.logger.info("Using Client-wise Random masking strategy.")
            return ClientwiseRandomMaskingStrategy(*common_args)
        else:
            raise ValueError(f"Unsupported SSFL mode: {mode}")

    # ===================================================================================
    #                               TRAINING ORCHESTRATION
    # ===================================================================================
    def train(self):
        self.logger.info(" Starting SSFL training process...")
        current_client_models_list = [copy.deepcopy(self.global_model_w) for _ in range(self.args.training.client_num_in_total)]

        # Initial mask generation for all modes
        self.masking_strategy.update_masks_if_needed(
            round_idx=0, 
            global_model_w=self.global_model_w, 
            clients=self.client_list, 
            prunable_layers=self.model_trainer.prunable_parameter_names,
            max_workers=self._max_workers,
        )
        # Update the reference in case the mask was created
        self.global_masks = self.masking_strategy.global_masks 

        # Apply the initial mask to the global model BEFORE training starts
        self.logger.info("Applying initial mask to the global model before starting training rounds.")
        self._apply_mask_to_global_model()
        
        for round_idx in range(1, self.args.training.comm_round): # Start from round 1
            self.logger.info(f"--- Communication Round {round_idx} / {self.args.training.comm_round} ---")

            local_weights_for_agg = self.masking_strategy.update_masks_if_needed(
                round_idx, self.global_model_w, self.client_list, 
                self.model_trainer.prunable_parameter_names, max_workers=self._max_workers,
            )
            self.global_masks = self.masking_strategy.global_masks

            # This variable will hold the average LR for the round
            avg_round_lr = 0.0

            # If the strategy didn't run a training round, run a standard one
            if not local_weights_for_agg:
                local_weights_for_agg, avg_round_lr = self._run_local_training_round(round_idx, current_client_models_list)

            if local_weights_for_agg:
                self.global_model_w = self._aggregate(local_weights_for_agg, mode=self.averaging_mode)
                self._apply_mask_to_global_model()
            
            self._evaluate_and_log_round(round_idx, current_client_models_list, avg_round_lr)

            torch.cuda.empty_cache()

        self.logger.info("SSFL training process finished.")
        self.save_checkpoint(self.global_model_w, current_client_models_list)

    # ===================================================================================
    #                               HELPER METHODS
    # ===================================================================================

    def _apply_mask_to_global_model(self):
        """
        Applies the global mask directly to the global_model_w state dictionary
        to enforce sparsity after aggregation.
        """
        if not self.global_masks:
            self.logger.warning("No global mask available to apply.")
            return

        self.logger.info("Applying global mask to the aggregated global model...")
        with torch.no_grad():
            for name, mask_tensor in self.global_masks.items():
                # global_model_w contains standard .weight keys
                if name in self.global_model_w:
                    # Move mask to the same device as the weight before multiplying
                    device = self.global_model_w[name].device
                    self.global_model_w[name] *= mask_tensor.to(device)


    def _train_single_client(self, client, round_idx):
        """
        Train a single client for one round.
        
        Args:
            client: Client instance to train
            round_idx: Current communication round
            
        Returns:
            TrainingResult: Results including weights, FLOPs, loss, sparsity
        """
        # Check if we're using client-wise masks
        if hasattr(self.masking_strategy, 'client_masks') and self.masking_strategy.client_masks:
            # Use the client-specific mask
            masks = self.masking_strategy.get_mask_for_client(client.client_idx)
        else:
            # Use the global mask (or None)
            masks = self.global_masks
        
        # This call now works with the simplified and corrected client.train
        result = client.train(
            global_model_w=copy.deepcopy(self.global_model_w), 
            round_num=round_idx, 
            masks=masks
        )
        return result



    def _run_local_training_round(self, round_idx, local_weights_models):
        """
        Sample clients and run parallel local training for one round.
        
        Args:
            round_idx: Current communication round
            local_weights_models: List to store updated local model weights
            
        Returns:
            tuple: (local_weights_for_agg, avg_learning_rate)
        """
        local_weights_for_agg = []
        learning_rates = []
        selected_clients = self._client_sampling(round_idx=round_idx)
        self.logger.info(f"Starting parallel training for {len(selected_clients)} clients...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            
            # Submit the helper function to the executor
            future_to_client = {
                executor.submit(self._train_single_client, client, round_idx): client 
                for client in selected_clients
            }
            
            # Process the results as they are completed
            for future in concurrent.futures.as_completed(future_to_client):
                client = future_to_client[future]
                try:
                    # Unpack the results returned by the helper function
                    result = future.result()
                    local_weights, flops, comm_params, loss, sparsity_dict, lr = (
                        result.weights, result.training_flops, result.comm_params, 
                        result.avg_loss, result.sparsity_dict, result.learning_rate
                    )
                    
                    self.tracker.add_client_stats(loss=loss, flops=flops, comm_params=comm_params)
                    local_weights_for_agg.append((client.get_sample_number(), local_weights))
                    local_weights_models[client.client_idx] = copy.deepcopy(local_weights)
                    learning_rates.append(lr)

                    # Log the client-specific sparsity to W&B
                    if self.args.wandb.mode == "online" and sparsity_dict:
                        # Only log overall sparsity, not layer-wise
                        overall_sparsity = sparsity_dict.get("sparsity_actual/overall")
                        if overall_sparsity is not None:
                            client_log = {f"client_sparsities/client_{client.client_idx}/overall": overall_sparsity}
                            wandb.log(client_log, step=round_idx)


                except Exception as exc:
                    self.logger.error(f"Client {client.client_idx} generated an exception during training: {exc}")
                    raise exc
        avg_lr = np.mean(learning_rates) if learning_rates else 0
        self.logger.info("...Parallel training round finished.")
        return local_weights_for_agg, avg_lr

    def _evaluate_and_log_round(self, round_idx, current_client_models_list, avg_lr):
        """
        Evaluate all clients and log metrics for the current round.
        
        Args:
            round_idx: Current communication round
            current_client_models_list: List of current local model weights
            avg_lr: Average learning rate for this round
        """
        if (round_idx + 1) % self.args.training.frequency_of_the_test == 0:
            prunable_layers = self.model_trainer.prunable_parameter_names
            actual_sparsity_dict = calculate_model_sparsity_from_weights(
                self.global_model_w, prunable_layers
            )
            
            # --- Pass all round info to the tracker ---
            num_clients = self.args.training.client_num_per_round
            self.tracker.set_round_info(sparsity=actual_sparsity_dict, num_clients=num_clients, learning_rate=avg_lr)
            
            # --- The rest of the evaluation and logging proceeds as before ---
            test_results = self._test_on_all_clients(current_client_models_list, round_idx)
            
            client_ids = list(range(self.args.training.client_num_in_total))
            self.tracker.set_test_metrics(test_results, client_ids=client_ids)
            
            self.tracker.log_round_to_wandb(round_idx)

    def _client_sampling(self, round_idx: int = None) -> List[Client]:
        """Sample clients for the current round."""
        num_clients_to_sample = self.args.training.client_num_per_round
        if num_clients_to_sample > len(self.client_list):
            return self.client_list
        return random.sample(self.client_list, num_clients_to_sample)
    

    def init_stat_info(self):
        self.stat_info = {
            "sum_comm_params": 0, "sum_training_flops": 0,
            "global_test_acc": [], "person_test_acc": []
        }

    # --- Aggregation Methods ---
    def _aggregate(self, w_locals, mode=None):
        if mode not in self.aggregation_strategies:
            raise ValueError(f"Unknown aggregation mode: {mode}")
        return self.aggregation_strategies[mode](w_locals)
    
    def _aggregate_fedavg(self, w_locals):
        training_num = sum([sample_num for sample_num, _ in w_locals])
        global_model_w = OrderedDict()
        
        # Move the first client's weights to the GPU to get device info
        first_client_weights = w_locals[0][1]
        device = next(iter(first_client_weights.values())).device

        # Pre-compute weights tensor once
        sample_weights = torch.tensor([num / training_num for num, _ in w_locals], device=device)
        
        for k in first_client_weights.keys():
            # Stack all tensors for the current layer 'k' into a new dimension
            # Shape becomes: (num_clients, original_shape...)
            layer_tensors = torch.stack([local_model_params[k] for _, local_model_params in w_locals])
            
            # Create a weight vector for the weighted average
            weights = sample_weights.view([-1] + [1] * (layer_tensors.dim() - 1))
            
            # Perform the weighted sum in a single, fast GPU operation
            global_model_w[k] = torch.sum(layer_tensors * weights, dim=0)
            
        return global_model_w
    
    def _aggregate_topk(self, w_locals):
        training_num = sum([sample_num for sample_num, _ in w_locals])
        global_model_w = OrderedDict()
        thresholds = get_threshold_list(self.args.model.dense_ratio, w_locals, self.model_trainer.prunable_parameter_names)
        for k in w_locals[0][1].keys():
            global_model_w[k] = torch.zeros_like(w_locals[0][1][k])
            for i in range(len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                tensor = local_model_params[k]
                if k in self.model_trainer.prunable_parameter_names:
                    global_model_w[k] += torch.where(tensor > thresholds[i], tensor, torch.tensor(0.0)) * w
                else:
                    global_model_w[k] += tensor * w
        return global_model_w

    def _aggregate_gradients(self, client_grads: List[dict]) -> dict:
        """Averages the gradients collected from clients."""
        if not client_grads: return {}
        grad_global = copy.deepcopy(client_grads[0])
        for k in grad_global.keys():
            for i in range(1, len(client_grads)):
                grad_global[k] += client_grads[i][k]
            grad_global[k] /= len(client_grads)
        return grad_global

    def _test_single_client(self, client_idx, w_per_mdl):
        """Helper function to run local_test on one client."""
        client = self.client_list[client_idx]
        
        # Pass self.global_masks when testing the global model
        g_test = client.local_test(self.global_model_w, True, masks=self.global_masks)
        
        # For personalized testing, we don't pass a mask unless you have per-client masks
        p_test = client.local_test(w_per_mdl, True) 
        
        return g_test, p_test

    def _test_on_all_clients(self, current_client_models_list, round_idx):
            self.logger.info(f"Performing parallel tests for all clients at round {round_idx}")
            
            global_results = []
            local_results = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                future_to_client = {
                    executor.submit(self._test_single_client, i, current_client_models_list[i]): i 
                    for i in range(self.args.training.client_num_in_total)
                }

                for future in concurrent.futures.as_completed(future_to_client):
                    # Simply collect the raw result dictionaries from each client
                    g_test, p_test = future.result()
                    global_results.append(g_test)
                    local_results.append(p_test)

            # The function's only job is to return the raw data.
            # All calculations will be handled by the StatsTracker.
            return {'global_results': global_results, 'local_results': local_results}


    def save_checkpoint(self, global_model_w, local_model_list):
        """
        Saves the final checkpoint using the training utilities.
        """
        # Gather masking strategy information
        masking_strategy_info = {}
        if hasattr(self.masking_strategy, 'last_mask_update_round'):
            masking_strategy_info['last_mask_update_round'] = self.masking_strategy.last_mask_update_round
        if hasattr(self.masking_strategy, 'next_mask_update_round'):
            masking_strategy_info['next_mask_update_round'] = self.masking_strategy.next_mask_update_round
        
        return save_final_checkpoint(
            exp_name=self.args.wandb.exp_name,
            global_model_w=global_model_w,
            local_model_list=local_model_list,
            config=self.args,
            global_masks=self.global_masks,
            masking_strategy_info=masking_strategy_info,
            logger=self.logger
        )

    def save_intermediate_checkpoint(self, round_idx, global_model_w, local_model_list=None):
        """
        Saves intermediate checkpoints using the training utilities.
        """
        # Gather masking strategy information
        masking_strategy_info = {}
        if hasattr(self.masking_strategy, 'last_mask_update_round'):
            masking_strategy_info['last_mask_update_round'] = self.masking_strategy.last_mask_update_round
        if hasattr(self.masking_strategy, 'next_mask_update_round'):
            masking_strategy_info['next_mask_update_round'] = self.masking_strategy.next_mask_update_round
        
        save_intermediate_checkpoint(
            exp_name=self.args.wandb.exp_name,
            round_idx=round_idx,
            global_model_w=global_model_w,
            config=self.args,
            local_model_list=local_model_list,
            global_masks=self.global_masks,
            masking_strategy_info=masking_strategy_info,
            logger=self.logger,
            save_frequency=10  # Save every 10 rounds, or adjust as needed
        )