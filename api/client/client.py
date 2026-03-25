from conf.config_schema import Config

class Client:
    """
    Container for a client's data and state in federated learning.
    
    The Client class holds client-specific data and delegates algorithm-specific
    training logic to a strategy object (SSFLClientStrategy).
    
    Attributes:
        client_idx: Unique identifier for this client
        local_training_data: DataLoader for local training data
        local_test_data: DataLoader for local test data
        local_sample_number: Number of training samples
        strategy: Algorithm-specific strategy object (FuseClientStrategy, etc.)
    """
    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, 
                 args, device, model_trainer, logger, strategy_class):
        self.args = args
        self.logger = logger
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.device = device

        # The client now holds an instance of a specific strategy class
        self.strategy = strategy_class(self, args, device, model_trainer, logger)

    def get_sample_number(self):
        """Returns the number of training samples for this client."""
        return self.local_sample_number

    def train(self, global_model_w, round_num, masks):
        """
        Perform local training on this client's data.
        
        Args:
            global_model_w: Global model weights to initialize from
            round_num: Current communication round number
            masks: Sparse masks to apply during training (None for dense training)
            
        Returns:
            TrainingResult: Named tuple containing weights, gradients, FLOPs, etc.
        """
        return self.strategy.train(global_model_w, masks, round_num)

    def generate_saliency_scores(self, method='ssfl', iterations=1):
        """
        Generate saliency scores for mask creation (FUSE-specific).
        
        Args:
            method: Saliency calculation method ('ssfl', 'ssfl_aux', etc.)
            iterations: Number of batches to average saliency over
            
        Returns:
            dict: Layer-wise saliency scores
        """
        if hasattr(self.strategy, 'generate_saliency_scores'):
            return self.strategy.generate_saliency_scores(method, iterations)
        else:
            self.logger.error("generate_saliency_scores is not supported by the current client strategy.")
            return {}

    def local_test(self, model_weights, use_test_dataset, masks=None):
        """
        Evaluate model on local data.
        
        Args:
            model_weights: Model weights to evaluate
            use_test_dataset: If True, use test data; otherwise use training data
            masks: Optional sparse masks to apply during testing
            
        Returns:
            dict: Test metrics including accuracy, loss, per-class accuracy
        """
        dataset = self.local_test_data if use_test_dataset else self.local_training_data
        self.strategy.model_trainer.set_model_params(model_weights)

        if masks:
            self.strategy.model_trainer.model.apply_masks(masks)

        metrics = self.strategy.model_trainer.test(dataset, self.device, self.args)

        if masks:
            self.strategy.model_trainer.model.remove_pruning()

        return metrics