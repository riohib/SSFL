"""
Training utility functions for FUSE experiments.
Contains checkpoint saving, loading, and other training-related utilities.
"""
import os
import torch
from typing import Dict, List, Optional, Any
from conf.config_schema import Config


def create_experiment_directories(exp_name: str) -> Dict[str, str]:
    """
    Creates the standard directory structure for an experiment.
    
    Args:
        exp_name: Name of the experiment
        
    Returns:
        Dictionary mapping directory types to their paths
    """
    base_output_dir = os.path.join(os.getcwd(), 'outputs', exp_name)
    
    directories = {
        'base': base_output_dir,
        'checkpoints': os.path.join(base_output_dir, 'checkpoints'),
        'checkpoints_intermediate': os.path.join(base_output_dir, 'checkpoints', 'intermediate'),
        'masks': os.path.join(base_output_dir, 'masks'),
        'masks_intermediate': os.path.join(base_output_dir, 'masks', 'intermediate'),
        'logs': os.path.join(base_output_dir, 'logs'),
        'wandb': os.path.join(base_output_dir, 'wandb')
    }
    
    # Create all directories
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories


def save_final_checkpoint(
    exp_name: str,
    global_model_w: Dict[str, torch.Tensor],
    local_model_list: List[Dict[str, torch.Tensor]],
    config: Config,
    global_masks: Optional[Dict[str, torch.Tensor]] = None,
    masking_strategy_info: Optional[Dict[str, Any]] = None,
    logger = None
) -> str:
    """
    Saves the final checkpoint with model weights, masks, and metadata.
    
    Args:
        exp_name: Name of the experiment
        global_model_w: Global model state dictionary
        local_model_list: List of local model state dictionaries
        config: Configuration object
        global_masks: Dictionary of pruning masks (optional)
        masking_strategy_info: Additional masking strategy information (optional)
        logger: Logger instance for output messages
        
    Returns:
        Path to the base experiment directory
    """
    directories = create_experiment_directories(exp_name)
    
    # Save main checkpoint with model weights
    checkpoint_path = os.path.join(directories['checkpoints'], f"{exp_name}_final.pt")
    checkpoint_data = {
        'global_state_dict': global_model_w,
        'local_model_list': local_model_list,
        'config': config,
        'training_completed': True,
        'final_round': config.training.comm_round - 1
    }
    torch.save(checkpoint_data, checkpoint_path)
    
    if logger:
        logger.info(f"Model checkpoint saved to {checkpoint_path}")
    
    # Save masks if they exist
    if global_masks:
        _save_masks(
            masks_dir=directories['masks'],
            exp_name=exp_name,
            global_masks=global_masks,
            config=config,
            masking_strategy_info=masking_strategy_info,
            logger=logger,
            is_final=True
        )
    
    # Save experiment metadata
    _save_experiment_metadata(
        base_dir=directories['base'],
        exp_name=exp_name,
        config=config,
        logger=logger
    )
    
    if logger:
        logger.info(f"All experiment outputs organized in: {directories['base']}")
    
    return directories['base']


def save_intermediate_checkpoint(
    exp_name: str,
    round_idx: int,
    global_model_w: Dict[str, torch.Tensor],
    config: Config,
    local_model_list: Optional[List[Dict[str, torch.Tensor]]] = None,
    global_masks: Optional[Dict[str, torch.Tensor]] = None,
    masking_strategy_info: Optional[Dict[str, Any]] = None,
    logger = None,
    save_frequency: int = 10
) -> None:
    """
    Saves intermediate checkpoints during training.
    
    Args:
        exp_name: Name of the experiment
        round_idx: Current training round
        global_model_w: Global model state dictionary
        config: Configuration object
        local_model_list: List of local model state dictionaries (optional)
        global_masks: Dictionary of pruning masks (optional)
        masking_strategy_info: Additional masking strategy information (optional)
        logger: Logger instance for output messages
        save_frequency: Save every N rounds
    """
    # Check if we should save this round
    should_save = (
        round_idx % save_frequency == 0 or
        (masking_strategy_info and 
         masking_strategy_info.get('last_mask_update_round') == round_idx)
    )
    
    if not should_save:
        return
    
    directories = create_experiment_directories(exp_name)
    
    # Save model checkpoint
    checkpoint_path = os.path.join(
        directories['checkpoints_intermediate'], 
        f"round_{round_idx:03d}.pt"
    )
    checkpoint_data = {
        'global_state_dict': global_model_w,
        'round': round_idx,
        'config': config
    }
    if local_model_list is not None:
        checkpoint_data['local_model_list'] = local_model_list
        
    torch.save(checkpoint_data, checkpoint_path)
    
    # Save masks if they exist
    if global_masks:
        masks_path = os.path.join(
            directories['masks_intermediate'], 
            f"masks_round_{round_idx:03d}.pt"
        )
        mask_data = {
            'global_masks': global_masks,
            'round': round_idx,
            'masking_strategy': config.algorithm.params.mode,
            'method': config.algorithm.params.method
        }
        if masking_strategy_info:
            mask_data.update(masking_strategy_info)
            
        torch.save(mask_data, masks_path)
        
    if logger:
        logger.info(f"Intermediate checkpoint saved for round {round_idx}")


def load_checkpoint(checkpoint_path: str, device: str = 'cpu') -> Dict[str, Any]:
    """
    Loads a checkpoint from the given path.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint on
        
    Returns:
        Dictionary containing the checkpoint data
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def load_masks(masks_path: str, device: str = 'cpu') -> Dict[str, Any]:
    """
    Loads masks from the given path.
    
    Args:
        masks_path: Path to the masks file
        device: Device to load the masks on
        
    Returns:
        Dictionary containing the masks data
    """
    if not os.path.exists(masks_path):
        raise FileNotFoundError(f"Masks not found at {masks_path}")
    
    masks = torch.load(masks_path, map_location=device)
    return masks


def _save_masks(
    masks_dir: str,
    exp_name: str,
    global_masks: Dict[str, torch.Tensor],
    config: Config,
    masking_strategy_info: Optional[Dict[str, Any]] = None,
    logger = None,
    is_final: bool = True
) -> None:
    """
    Internal function to save masks and generate human-readable summary.
    """
    # Save masks as PyTorch file
    suffix = "final" if is_final else "intermediate"
    masks_path = os.path.join(masks_dir, f"{exp_name}_{suffix}_masks.pt")
    
    mask_data = {
        'global_masks': global_masks,
        'masking_strategy': config.algorithm.params.mode,
        'method': config.algorithm.params.method,
        'dense_ratio': config.model.dense_ratio,
    }
    
    if masking_strategy_info:
        mask_data.update(masking_strategy_info)
    
    torch.save(mask_data, masks_path)
    
    if logger:
        logger.info(f"Masks saved to {masks_path}")
    
    # Generate human-readable summary
    if is_final:
        masks_txt_path = os.path.join(masks_dir, f"{exp_name}_mask_summary.txt")
        with open(masks_txt_path, 'w') as f:
            f.write(f"Mask Summary for Experiment: {exp_name}\n")
            f.write(f"Masking Strategy: {config.algorithm.params.mode}\n")
            f.write(f"Method: {config.algorithm.params.method}\n")
            f.write(f"Target Dense Ratio: {config.model.dense_ratio}\n\n")
            
            total_params = 0
            total_active = 0
            
            for layer_name, mask in global_masks.items():
                density = mask.float().mean().item()
                layer_total = mask.numel()
                layer_active = mask.sum().item()
                
                total_params += layer_total
                total_active += layer_active
                
                f.write(f"{layer_name}:\n")
                f.write(f"  Total Parameters: {layer_total:,}\n")
                f.write(f"  Active Parameters: {layer_active:,}\n")
                f.write(f"  Density: {density:.4f} ({density*100:.2f}%)\n")
                f.write(f"  Sparsity: {1-density:.4f} ({(1-density)*100:.2f}%)\n\n")
            
            # Overall summary
            overall_density = total_active / total_params if total_params > 0 else 0
            f.write(f"OVERALL SUMMARY:\n")
            f.write(f"  Total Parameters: {total_params:,}\n")
            f.write(f"  Active Parameters: {total_active:,}\n")
            f.write(f"  Overall Density: {overall_density:.4f} ({overall_density*100:.2f}%)\n")
            f.write(f"  Overall Sparsity: {1-overall_density:.4f} ({(1-overall_density)*100:.2f}%)\n")


def _save_experiment_metadata(
    base_dir: str,
    exp_name: str,
    config: Config,
    logger = None
) -> None:
    """
    Internal function to save experiment metadata.
    """
    metadata_path = os.path.join(base_dir, f"{exp_name}_metadata.txt")
    
    with open(metadata_path, 'w') as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Mode: {config.algorithm.params}\n")
        f.write(f"Method: {config.algorithm.params}\n")
        f.write(f"Dataset: {config.dataset.name}\n")
        f.write(f"Model: {config.model.name}\n")
        f.write(f"Communication Rounds: {config.training.comm_round}\n")
        f.write(f"Clients: {config.training.client_num_in_total}\n")
        f.write(f"Target Dense Ratio: {config.model.dense_ratio}\n")
        
        if hasattr(config.algorithm.params, 'mask_freq_growth_factor') and config.algorithm.params.mask_freq_growth_factor:
            f.write(f"Mask Frequency Growth Factor: {config.algorithm.params.mask_freq_growth_factor}\n")
        if hasattr(config.algorithm.params, 'initial_mask_freq'):
            f.write(f"Initial Mask Frequency: {config.algorithm.params.initial_mask_freq}\n")
        if hasattr(config.algorithm.params, 'prune_grow_ratio') and config.algorithm.params.prune_grow_ratio:
            f.write(f"Prune-Grow Ratio: {config.algorithm.params.prune_grow_ratio}\n")
            
        f.write(f"\nDirectory Structure:\n")
        f.write(f"├── checkpoints/           # Model weights and states\n")
        f.write(f"│   └── intermediate/      # Round-by-round checkpoints\n")
        f.write(f"├── masks/                 # Pruning masks and summaries\n")
        f.write(f"│   └── intermediate/      # Round-by-round masks\n")
        f.write(f"├── logs/                  # Training logs\n")
        f.write(f"├── wandb/                 # Weights & Biases data\n")
        f.write(f"└── {exp_name}_metadata.txt\n")
    
    if logger:
        logger.info(f"Experiment metadata saved to {metadata_path}")


def get_latest_checkpoint(exp_name: str, checkpoint_type: str = 'final') -> Optional[str]:
    """
    Gets the path to the latest checkpoint for an experiment.
    
    Args:
        exp_name: Name of the experiment
        checkpoint_type: 'final' or 'intermediate'
        
    Returns:
        Path to the latest checkpoint, or None if not found
    """
    base_dir = os.path.join(os.getcwd(), 'outputs', exp_name, 'checkpoints')
    
    if checkpoint_type == 'final':
        final_path = os.path.join(base_dir, f"{exp_name}_final.pt")
        return final_path if os.path.exists(final_path) else None
    
    elif checkpoint_type == 'intermediate':
        intermediate_dir = os.path.join(base_dir, 'intermediate')
        if not os.path.exists(intermediate_dir):
            return None
        
        # Find the latest round checkpoint
        checkpoints = [f for f in os.listdir(intermediate_dir) if f.startswith('round_') and f.endswith('.pt')]
        if not checkpoints:
            return None
        
        # Sort by round number
        checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        return os.path.join(intermediate_dir, checkpoints[-1])
    
    return None