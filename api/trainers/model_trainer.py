import copy
import numpy as np
import torch
from torch import nn

from api.trainers.base_trainer import BaseModelTrainer
from conf.config_schema import Config
import torch.optim.lr_scheduler as lr_scheduler

class ModelTrainer(BaseModelTrainer):
    def __init__(self, model, args=None, logger = None):
        super().__init__(model, args)
        self.args=args
        self.logger = logger
        self.model = model

    @property
    def prunable_parameter_names(self) -> list:
        """
        Returns a list of plain parameter names (e.g., 'conv1.weight')
        that are considered prunable. This is the single source of truth.
        """
        prunable_names = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prunable_names.append(f"{name}.weight")
        return prunable_names

    def get_trainable_params(self):
        """
        Returns a dictionary of the model's trainable parameters.
        """
        params_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_dict[name] = param
        return params_dict

    # Returns a plain state_dict because self.model is a SparseModel
    def get_model_params(self):
        return self.model.cpu().state_dict()

    # Loads a plain state_dict because self.model is a SparseModel
    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)
    

    def screen_gradients(self, batch, device):
        model = self.model
        model.to(device)
        model.eval()
        criterion = nn.CrossEntropyLoss().to(device)

        model.zero_grad()
        (x, labels) = batch
        x, labels = x.to(device), labels.to(device)
        log_probs = model.forward(x)
        loss = criterion(log_probs, labels.long())
        loss.backward()
        
        gradient={}
        # Iterate over the underlying model's parameters to get prefixed names
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Store the gradient with the plain name
                gradient[name] = param.grad.to("cpu")
        return gradient
    

    def train(self, train_data, device, args: Config, round, masks, calculate_grads=False):
        model = self.model
        model.to(device)
        model.train()
        # train and update
        criterion = nn.CrossEntropyLoss().to(device)

        if args.optimizer.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.optimizer.lr,
                momentum=args.optimizer.momentum,
                weight_decay=args.optimizer.wd
            )
        
        # Select and apply the LR scheduling strategy
        scheduler_name = getattr(args.optimizer, 'scheduler', 'default')
        scheduler = None
        self.logger.info(f"Using LR scheduler strategy: '{scheduler_name}'")

        if scheduler_name == 'default':
            current_lr = args.optimizer.lr * (args.optimizer.lr_decay**round)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            self.logger.info(f"Applied default round-based decay. Current LR: {current_lr:.6f}")

        elif scheduler_name == 'cosine_annealing':
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=args.optimizer.scheduler_cycle_len,
                T_mult=1,
                eta_min=1e-5
            )
        else:
            raise ValueError(f"Unsupported scheduler specified in config: '{scheduler_name}'")
        
        total_loss = 0
        total_batches = 0
        
        
        for epoch in range(args.training.epochs):
            epoch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model.forward(x)
                loss = criterion(log_probs, labels.long())
                loss.backward()
                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                optimizer.step()
                epoch_loss.append(loss.item())

                total_loss += loss.item()
                total_batches += 1
            
            epoch_avg_loss = sum(epoch_loss) / len(epoch_loss) if len(epoch_loss) > 0 else 0
            
            if scheduler is not None:
                scheduler.step()

            if hasattr(self, 'id'):
                current_lr = optimizer.param_groups[0]['lr']
                self.logger.info(f"Client Index = {self.id}\tEpoch: {epoch+1}\t"
                                f"Loss: {epoch_avg_loss:.6f}\tLR: {current_lr:.6f}")
            
        final_grads = None

        if calculate_grads:
            self.logger.info(f"Client {getattr(self, 'id', 'N/A')}: Performing full gradient calculation.")
            model.zero_grad()
            for x, labels in train_data:
                x, labels = x.to(device), labels.to(device)
                log_probs = model.forward(x)
                loss = criterion(log_probs, labels.long())
                loss = loss / len(train_data) 
                loss.backward()
            final_grads = {name: param.grad.cpu().clone() 
               for name, param in model.named_parameters() if param.grad is not None}

        
        overall_avg_loss = total_loss / total_batches if total_batches > 0 else 0
        final_lr = optimizer.param_groups[0]['lr']
        return overall_avg_loss, final_grads, final_lr


    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_acc':0.0,
            'test_loss': 0,
            'test_total': 0,
            'per_class_correct': {},  # Dict mapping class_id -> count of correct predictions
            'per_class_total': {}     # Dict mapping class_id -> total samples
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target.long())

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
                
                # Track per-class accuracy
                for class_id in range(pred.size(1)):  # pred.size(1) is num_classes
                    class_mask = (target == class_id)
                    if class_mask.any():
                        class_correct = (predicted[class_mask] == target[class_mask]).sum().item()
                        class_total = class_mask.sum().item()
                        
                        if class_id not in metrics['per_class_correct']:
                            metrics['per_class_correct'][class_id] = 0
                            metrics['per_class_total'][class_id] = 0
                        
                        metrics['per_class_correct'][class_id] += class_correct
                        metrics['per_class_total'][class_id] += class_total
                
                metrics['test_acc'] = metrics['test_correct'] / metrics['test_total']
        
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

