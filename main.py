import logging
import os
import random
import time
import sys

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

# Use absolute paths from your project root
from conf.config_loader import load_app_config
from conf.config_schema import Config
from api.model.resnet import customized_resnet18, customized_resnet50
from api.model.sparse_model import SparseModel
from api.trainers.model_trainer import ModelTrainer

from api.algorithms.ssfl_runner import SSFL


def load_data(cfg: Config, logger):
    start = time.time()
    
    if cfg.dataset.name == "cifar10":
        from data_preprocessing.cifar10_partitioner import load_and_partition_cifar10
        dataset = load_and_partition_cifar10(
            data_dir=cfg.dataset.data_dir,
            partition_method=cfg.dataset.partition_method,
            n_clients=cfg.training.client_num_in_total,
            partition_alpha=cfg.dataset.partition_alpha,
            batch_size=cfg.training.batch_size,
            max_workers=cfg.dataset.data_loader_max_workers,
            logger=logger
        )
    elif cfg.dataset.name == "cifar100":
        from data_preprocessing.cifar100_partitioner import load_and_partition_cifar100
        dataset = load_and_partition_cifar100(
            data_dir=cfg.dataset.data_dir,
            partition_method=cfg.dataset.partition_method,
            n_clients=cfg.training.client_num_in_total,
            partition_alpha=cfg.dataset.partition_alpha,
            batch_size=cfg.training.batch_size,
            max_workers=cfg.dataset.data_loader_max_workers,
            logger=logger
        )
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")
        
    end = time.time()
    logger.info(f"Data loading completed in {end - start:.2f} seconds")
    return dataset



def create_model(cfg: Config, class_num):
    """
    Create model based on configuration.
    
    Supported models:
    - resnet18: ResNet-18 for CIFAR-10/100
    - resnet50: ResNet-50 for CIFAR-10/100
    """
    model = None

    if cfg.model.name == "resnet18":
        model = customized_resnet18(class_num=class_num)
    elif cfg.model.name == "resnet50":
        model = customized_resnet50(class_num=class_num)
    else:
        raise ValueError(f"Unsupported model: {cfg.model.name}. Supported models: resnet18, resnet50")

    # Wrap the final model in SparseModel class
    return SparseModel(model)


def custom_model_trainer(cfg: Config, model, logger):
    return ModelTrainer(model, cfg, logger)


def logger_config(exp_name, logging_name):
    """
    Creates a logger that writes to the experiment's output directory
    instead of a separate logs directory.
    """
    # Create the experiment output directory structure
    base_output_dir = os.path.join(os.getcwd(), 'outputs', exp_name)
    logs_dir = os.path.join(base_output_dir, 'logs')
    
    # Create directories if they don't exist
    os.makedirs(logs_dir, exist_ok=True)
    
    # Log file path within the experiment directory
    log_file_path = os.path.join(logs_dir, f'{exp_name}.log')
    
    logger = logging.getLogger(logging_name)
    # Set to INFO for production; can be overridden by config.experiment.debug_mode
    logger.setLevel(level=logging.INFO if not hasattr(logging, 'debug_mode') else logging.DEBUG)

    # Clear any existing handlers to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- Handler 1: Writes INFO messages and up to a file ---
    file_handler = logging.FileHandler(log_file_path, mode='w+', encoding='UTF-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # --- Handler 2: Writes INFO messages and up to the terminal ---
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    # Log where the log file is being saved
    logger.info(f"Logging to file: {log_file_path}")
    
    return logger

def wandb_init(cfg: Config):
    """
    Initialize wandb with experiment-specific directory for offline storage.
    """
    # Create wandb directory within experiment output
    base_output_dir = os.path.join(os.getcwd(), 'outputs', cfg.wandb.exp_name)
    wandb_dir = os.path.join(base_output_dir, 'wandb')
    os.makedirs(wandb_dir, exist_ok=True)
    
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.exp_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wandb.mode,
        dir=wandb_dir,  # This directs wandb to use our experiment directory
    )


def main() -> None:
    # 1. Get the configuration object by calling our explicit loader
    cfg = load_app_config()

    # The rest of the program proceeds as before
    OmegaConf.set_struct(cfg, False)

    wandb_init(cfg)
    logger = logger_config(cfg.wandb.exp_name, cfg.wandb.exp_name)
    
    if cfg.wandb.mode == "online":
        logger.info(f"Wandb run URL: {wandb.run.url}")
    else:
        logger.info("Wandb is in disabled/offline mode.")
    
    logger.info(f"Running algorithm: {cfg.algorithm.name.upper()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    device = torch.device(f"cuda:{cfg.experiment.gpu}" if torch.cuda.is_available() else "cpu")

    # Calculate derived parameter
    cfg.training.client_num_per_round = int(cfg.training.client_num_in_total * cfg.training.frac)

    logger.info(OmegaConf.to_yaml(cfg))
    logger.info(device)

    # Set random seeds
    random.seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)
    torch.manual_seed(cfg.experiment.seed)
    torch.cuda.manual_seed_all(cfg.experiment.seed)
    torch.backends.cudnn.deterministic = True

    # Load data
    dataset = load_data(cfg, logger)

    # Create model
    model = create_model(cfg, class_num=len(dataset.traindata_cls_counts[0]))
    
    # Create model trainer
    model_trainer = ModelTrainer(model, cfg, logger)
    logger.info(model)

    # Instantiate and run SSFL
    ssfl = SSFL(dataset, device, cfg, model_trainer, logger)
    ssfl.train()

    if cfg.wandb.mode == "online":
        wandb.finish()

if __name__ == "__main__":
    main()
