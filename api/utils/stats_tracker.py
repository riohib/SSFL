import numpy as np
import torch
import wandb

class StatsTracker:
    """A dedicated class to handle collecting and logging experiment statistics."""

    def __init__(self, wandb_mode, logger, ood_enabled=False, ood_client_ids=None, ood_classes=None):
        self.wandb_mode = wandb_mode
        self.logger = logger
        self.ood_enabled = ood_enabled
        self.ood_client_ids = ood_client_ids if ood_client_ids else set()
        self.ood_classes = ood_classes if ood_classes else []
        self.round_lr = 0.0
        self._reset_round_metrics()
        
        # Define OOD metrics with descriptions for WandB
        if self.ood_enabled:
            self._define_ood_metrics()

    def _reset_round_metrics(self):
        """Resets the stored metrics at the end of a round."""
        self.round_losses = []
        self.round_training_flops = 0
        self.round_comm_params = 0
        self.round_flip_rate = None
        self.global_test_metrics = 0
        self.local_acc_list = []
        self.round_sparsity = {}
        
        # OOD-specific metrics
        self.ood_metrics = {
            'global_ood_acc': 0.0,
            'global_id_acc': 0.0,
            'global_per_class_acc': {},
            'local_ood_clients_acc': [],
            'local_id_clients_acc': [],
            'local_per_class_acc': {}
        }
        
        # Mask shift metrics for OOD experiments
        self.mask_jaccard_distance = None

    def add_client_stats(self, loss, flops, comm_params):
        """Adds the stats from a single client's training round."""
        self.round_losses.append(loss)
        self.round_training_flops += flops
        self.round_comm_params += comm_params

    def calculate_and_store_flip_rate(self, old_mask, new_mask):
        """Calculates the percentage of weights that flipped in the mask."""
        if old_mask is None or new_mask is None:
            return

        total_weights = 0
        flipped_weights = 0
        for name in new_mask:
            if name in old_mask:
                total_weights += new_mask[name].numel()
                flipped_weights += torch.sum(new_mask[name] != old_mask[name])

        if total_weights > 0:
            self.round_flip_rate = (flipped_weights / total_weights) * 100
    
    def calculate_jaccard_distance(self, old_mask, new_mask):
        """
        Calculates Jaccard distance between two masks.
        Jaccard distance = 1 - Jaccard similarity
        Jaccard similarity = |A ∩ B| / |A ∪ B|
        
        Returns:
            Jaccard distance (float) or None if masks are invalid
        """
        if old_mask is None or new_mask is None:
            return None
        
        intersection = 0
        union = 0
        
        for name in new_mask:
            if name in old_mask:
                old_m = old_mask[name].bool()
                new_m = new_mask[name].bool()
                
                # Intersection: weights that are 1 in both masks
                intersection += torch.sum(old_m & new_m).item()
                
                # Union: weights that are 1 in either mask
                union += torch.sum(old_m | new_m).item()
        
        if union == 0:
            return None
        
        jaccard_similarity = intersection / union
        jaccard_distance = 1.0 - jaccard_similarity
        
        return jaccard_distance

    def set_test_metrics(self, test_results: dict, client_ids: list = None):
            """
            Processes lists of raw test results from all clients to calculate
            final aggregate and distributive metrics.
            
            Args:
                test_results: Dict with 'global_results' and 'local_results' lists
                client_ids: List of client IDs corresponding to results (for OOD analysis)
            """
            global_results = test_results.get('global_results', [])
            local_results = test_results.get('local_results', [])
            
            if client_ids is None:
                client_ids = list(range(len(global_results)))

            # --- Process Global Model Results ---
            if global_results:
                total_samples = sum(res['test_total'] for res in global_results)
                total_correct = sum(res['test_correct'] for res in global_results)
                self.global_test_metrics = (total_correct / total_samples) if total_samples > 0 else 0
                self.logger.info(f"Global Model Accuracy: {self.global_test_metrics:.4f}")
                
                # Process OOD-specific metrics for global model
                if self.ood_enabled:
                    self._process_ood_metrics_global(global_results, client_ids)

            # --- Process Personalized Model Results ---
            if local_results:
                # Store the full list of individual accuracies for distribution stats
                self.local_acc_list = [res['test_acc'] for res in local_results]
                # You can log the average here if you want to see it in the console
                avg_local_acc = np.mean(self.local_acc_list) if self.local_acc_list else 0
                self.logger.info(f"Avg Personalized Model Accuracy: {avg_local_acc:.4f}")
                
                # Process OOD-specific metrics for local models
                if self.ood_enabled:
                    self._process_ood_metrics_local(local_results, client_ids)
    
    def _process_ood_metrics_global(self, global_results: list, client_ids: list):
        """Process OOD-specific metrics for global model."""
        # Aggregate per-class accuracy across all clients
        per_class_correct = {}
        per_class_total = {}
        
        for res in global_results:
            if 'per_class_correct' in res and 'per_class_total' in res:
                for class_id, correct_count in res['per_class_correct'].items():
                    if class_id not in per_class_correct:
                        per_class_correct[class_id] = 0
                        per_class_total[class_id] = 0
                    per_class_correct[class_id] += correct_count
                    per_class_total[class_id] += res['per_class_total'].get(class_id, 0)
        
        # Calculate per-class accuracy
        for class_id in per_class_total:
            if per_class_total[class_id] > 0:
                self.ood_metrics['global_per_class_acc'][class_id] = (
                    per_class_correct[class_id] / per_class_total[class_id]
                )
        
        # Calculate OOD vs ID class accuracy
        ood_correct = sum(per_class_correct.get(c, 0) for c in self.ood_classes)
        ood_total = sum(per_class_total.get(c, 0) for c in self.ood_classes)
        id_classes = [c for c in range(10) if c not in self.ood_classes]  # Assuming 10 classes for CIFAR-10
        id_correct = sum(per_class_correct.get(c, 0) for c in id_classes)
        id_total = sum(per_class_total.get(c, 0) for c in id_classes)
        
        self.ood_metrics['global_ood_acc'] = (ood_correct / ood_total) if ood_total > 0 else 0.0
        self.ood_metrics['global_id_acc'] = (id_correct / id_total) if id_total > 0 else 0.0
        
        if self.logger:
            self.logger.info(f"Global OOD Classes ({self.ood_classes}) Accuracy: {self.ood_metrics['global_ood_acc']:.4f}")
            self.logger.info(f"Global ID Classes Accuracy: {self.ood_metrics['global_id_acc']:.4f}")
    
    def _process_ood_metrics_local(self, local_results: list, client_ids: list):
        """Process OOD-specific metrics for local/personalized models."""
        ood_accs = []
        id_accs = []
        
        for idx, res in enumerate(local_results):
            client_id = client_ids[idx] if idx < len(client_ids) else idx
            
            if client_id in self.ood_client_ids:
                ood_accs.append(res['test_acc'])
            else:
                id_accs.append(res['test_acc'])
        
        self.ood_metrics['local_ood_clients_acc'] = ood_accs
        self.ood_metrics['local_id_clients_acc'] = id_accs
        
        if self.logger:
            if ood_accs:
                self.logger.info(f"Avg OOD Clients Accuracy: {np.mean(ood_accs):.4f} (n={len(ood_accs)})")
            if id_accs:
                self.logger.info(f"Avg ID Clients Accuracy: {np.mean(id_accs):.4f} (n={len(id_accs)})")


    def _define_ood_metrics(self):
        """Define OOD metrics with descriptions for WandB."""
        if self.wandb_mode != "online":
            return
        
        try:
            import wandb
            
            # Define metrics with summary statistics
            wandb.define_metric("ood/global_ood_classes_acc", 
                               summary="mean",
                               goal="maximize",
                               _step="round")
            wandb.define_metric("ood/global_id_classes_acc",
                               summary="mean", 
                               goal="maximize",
                               _step="round")
            wandb.define_metric("ood/local_ood_clients_acc_mean",
                               summary="mean",
                               goal="maximize", 
                               _step="round")
            wandb.define_metric("ood/local_id_clients_acc_mean",
                               summary="mean",
                               goal="maximize",
                               _step="round")
            
            # Define mask shift metrics
            wandb.define_metric("mask/jaccard_distance",
                               summary="mean",
                               goal="minimize",
                               _step="round")
            
            # Log metric descriptions as config (for documentation)
            # Note: WandB doesn't support descriptions directly in define_metric,
            # but we can add them to the config or create a summary table
            metric_descriptions = {
                "ood/global_ood_classes_acc": "Global model accuracy on OOD classes (e.g., 8-9) across all test data",
                "ood/global_id_classes_acc": "Global model accuracy on In-Distribution classes (e.g., 0-7) across all test data",
                "ood/per_class_acc/class_X": "Global model accuracy for each individual class (0-9)",
                "ood/local_ood_clients_acc_mean": "Average accuracy of personalized models on OOD clients",
                "ood/local_ood_clients_acc_std": "Standard deviation of accuracy across OOD clients",
                "ood/local_id_clients_acc_mean": "Average accuracy of personalized models on ID clients",
                "ood/local_id_clients_acc_std": "Standard deviation of accuracy across ID clients",
                "ood/partition_heatmap": "Class distribution heatmap showing sample counts per client-class",
                "ood/client_class_matrix": "Binary matrix showing which classes each client has",
                "mask/jaccard_distance": "Jaccard distance (1 - Jaccard similarity) between old and new masks. Measures mask shift during updates. Higher values indicate more mask change. Logged during mask updates, especially at OOD introduction."
            }
            
            # Store descriptions in wandb config for reference
            # Only update if wandb is already initialized
            try:
                if wandb.run is not None and hasattr(wandb, 'config'):
                    wandb.config.update({"ood_metric_descriptions": metric_descriptions})
            except:
                # WandB not initialized yet, that's okay - descriptions will be in artifact
                pass
                
        except Exception as e:
            # If wandb is not initialized yet, that's okay - we'll skip this
            pass

    def set_round_info(self, sparsity: dict, num_clients: int, learning_rate: float):
        """Sets general, round-specific information."""
        self.round_sparsity = sparsity
        self.num_selected_clients = num_clients
        self.round_lr = learning_rate

    def log_round_to_wandb(self, round_idx):
        """Aggregates round stats and logs them to Weights & Biases."""
        if self.wandb_mode != "online":
            return

        avg_loss = np.mean(self.round_losses) if self.round_losses else 0

        # 1. Create a dictionary with ONLY the scalar metrics
        scalar_log_dict = {
            "avg_local_loss": avg_loss,
            "global/test_accuracy": self.global_test_metrics,
            "global/total_training_flops": self.round_training_flops,
            "global/total_comm_params": self.round_comm_params,
            "global/learning_rate": self.round_lr,
        }

        if self.round_sparsity:
            scalar_log_dict.update(self.round_sparsity)

        if self.round_flip_rate is not None:
            scalar_log_dict["mask/flip_rate_percent"] = self.round_flip_rate
        
        # Log mask Jaccard distance if available (for OOD mask updates)
        if self.mask_jaccard_distance is not None:
            scalar_log_dict["mask/jaccard_distance"] = self.mask_jaccard_distance

        if self.local_acc_list:
            scalar_log_dict["local/acc_mean"] = np.mean(self.local_acc_list)
            scalar_log_dict["local/acc_std"] = np.std(self.local_acc_list)
            scalar_log_dict["local/acc_max"] = np.max(self.local_acc_list)
            scalar_log_dict["local/acc_min"] = np.min(self.local_acc_list)

        # 2. Log all scalar metrics in one call
        wandb.log(scalar_log_dict, step=round_idx)

        # 3. Log the histogram in a separate call
        if self.local_acc_list:
            wandb.log({"local/accuracy_histogram": wandb.Histogram(self.local_acc_list)}, step=round_idx)
        
        # 4. Log OOD-specific metrics
        if self.ood_enabled:
            ood_log_dict = {
                "ood/global_ood_classes_acc": self.ood_metrics['global_ood_acc'],
                "ood/global_id_classes_acc": self.ood_metrics['global_id_acc'],
            }
            
            # Per-class accuracy
            for class_id, acc in self.ood_metrics['global_per_class_acc'].items():
                ood_log_dict[f"ood/per_class_acc/class_{class_id}"] = acc
            
            # OOD vs ID client accuracy
            if self.ood_metrics['local_ood_clients_acc']:
                ood_log_dict["ood/local_ood_clients_acc_mean"] = np.mean(self.ood_metrics['local_ood_clients_acc'])
                ood_log_dict["ood/local_ood_clients_acc_std"] = np.std(self.ood_metrics['local_ood_clients_acc'])
                wandb.log({"ood/local_ood_clients_histogram": wandb.Histogram(self.ood_metrics['local_ood_clients_acc'])}, step=round_idx)
            
            if self.ood_metrics['local_id_clients_acc']:
                ood_log_dict["ood/local_id_clients_acc_mean"] = np.mean(self.ood_metrics['local_id_clients_acc'])
                ood_log_dict["ood/local_id_clients_acc_std"] = np.std(self.ood_metrics['local_id_clients_acc'])
                wandb.log({"ood/local_id_clients_histogram": wandb.Histogram(self.ood_metrics['local_id_clients_acc'])}, step=round_idx)
            
            # Log OOD metrics
            wandb.log(ood_log_dict, step=round_idx)
            
            # Create per-class accuracy table
            if self.ood_metrics['global_per_class_acc']:
                class_table_data = []
                for class_id in sorted(self.ood_metrics['global_per_class_acc'].keys()):
                    class_type = "OOD" if class_id in self.ood_classes else "ID"
                    class_table_data.append([
                        class_id,
                        class_type,
                        f"{self.ood_metrics['global_per_class_acc'][class_id]:.4f}"
                    ])
                
                class_table = wandb.Table(
                    columns=["Class", "Type", "Accuracy"],
                    data=class_table_data
                )
                wandb.log({"ood/per_class_accuracy_table": class_table}, step=round_idx)

        # Reset for the next round
        self._reset_round_metrics()