from dataclasses import dataclass, field
from typing import Any, Literal, Optional 

@dataclass
class SSFLConfig:
    """Parameters for the SSFL algorithm's masking strategies."""
    mode: str = "static"
    method: str = "ssfl"
    initial_mask_freq: int = 0
    warmup_rounds: int = 0  # Number of rounds to train with all-1's mask before pruning
    
    def __post_init__(self):
        """Validate SSFL-specific configuration after initialization."""
        valid_modes = ["static", "random_global", "random_layerwise", "random_clientwise"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid SSFL mode '{self.mode}'. Must be one of: {valid_modes}")
            
        valid_methods = ["ssfl", "ssfl_aux"]
        if self.method not in valid_methods:
            raise ValueError(f"Invalid SSFL method '{self.method}'. Must be one of: {valid_methods}")

@dataclass
class AlgorithmSSFLConfig:
    """The main configuration when 'ssfl' is the selected algorithm."""
    name: Literal["ssfl"] = "ssfl"
    params: SSFLConfig = field(default_factory=SSFLConfig)


@dataclass
class ModelConfig:
    name: str = "resnet18"
    dense_ratio: float = 0.2
    anneal_factor: float = 0.5
    cs: str = "v0"
    itersnip_iteration: int = 1
    erk_power_scale: float = 1.0

@dataclass
class DatasetConfig:
    name: str = "cifar10"  # Can be "cifar10" or "cifar100"
    data_dir: str = "./dataset"
    partition_method: str = "dir"
    partition_alpha: float = 0.3
    public_portion: float = 0.0
    data_loader_max_workers: Optional[int] = None  # None = use os.cpu_count(), set to 4 for RTX systems
    
    def __post_init__(self):
        """Validate dataset configuration after initialization"""
        valid_datasets = ["cifar10", "cifar100"]
        if self.name not in valid_datasets:
            raise ValueError(f"Invalid dataset '{self.name}'. Must be one of: {valid_datasets}")

@dataclass
class OptimizerConfig:
    """Configuration for the client-side optimizer."""
    client_optimizer: str = "sgd"
    lr: float = 0.1
    lr_decay: float = 0.998
    wd: float = 5e-4
    momentum: float = 0.0
    scheduler: str = "default"
    scheduler_cycle_len: int = 10 

@dataclass
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 5
    comm_round: int = 4
    frac: float = 0.5
    client_num_in_total: int = 4
    client_num_per_round: int = -1
    frequency_of_the_test: int = 1
    max_workers: int = -1  # -1 means use client_num_in_total


@dataclass
class ExperimentConfig:
    seed: int = 550
    gpu: int = 0
    ci: int = 0
    tag: str = "test"
    debug_mode: bool = False

@dataclass
class WandbConfig:
    mode: str = "online"
    project: str = "ssfl-experiments"
    exp_name: str = "test_exp1"


@dataclass
class Config:
    """The main configuration schema for the application."""
    
    algorithm: Any = field(default_factory=dict)

    # We revert to using the class name directly as the default_factory.
    # This is standard, simple, and avoids the NameError.
    optimizer: 'OptimizerConfig' = field(default_factory=OptimizerConfig)
    training: 'TrainingConfig' = field(default_factory=TrainingConfig)
    model: 'ModelConfig' = field(default_factory=ModelConfig)
    dataset: 'DatasetConfig' = field(default_factory=DatasetConfig)
    experiment: 'ExperimentConfig' = field(default_factory=ExperimentConfig)
    wandb: 'WandbConfig' = field(default_factory=WandbConfig)
