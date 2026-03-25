import torch
import copy
import numpy as np

def calculate_model_sparsity_from_weights(model_weights: dict, prunable_layers: list) -> dict:
    """
    Calculates overall and layer-wise sparsity directly from a model's state_dict,
    considering only the prunable layers.
    """
    layer_sparsities = {}
    total_zeros = 0
    total_weights = 0
    layers_found = 0 # <-- Add a counter

    for name, tensor in model_weights.items():
        if name in prunable_layers:
            layers_found += 1 # <-- Increment if a match is found
            # ... (rest of the loop is unchanged)
            layer_zeros = (tensor == 0).sum().item()
            layer_total = tensor.numel()
            if layer_total > 0:
                layer_sparsities[f"sparsity_actual/{name}"] = (layer_zeros / layer_total) * 100
            total_zeros += layer_zeros
            total_weights += layer_total

    # --- NEW: Add an assertion here ---
    # If we were given a list of layers to check but found none, something is wrong.
    if prunable_layers and layers_found == 0:
        raise AssertionError(
            "Sparsity calculation failed: None of the provided 'prunable_layers' were found in the "
            f"model's weights. This indicates a name mismatch.\n"
            f"  - Example Prunable Layer Name: '{prunable_layers[0]}'\n"
            f"  - Example Model Weight Key:    '{list(model_weights.keys())[0]}'"
        )

    if total_weights > 0:
        layer_sparsities["sparsity_actual/overall"] = (total_zeros / total_weights) * 100
    else:
        layer_sparsities["sparsity_actual/overall"] = 0

    return layer_sparsities


## TOPK TOOLS
def select_model_topk(k, model_w, prunable_layers):
    all_values = torch.cat([v.flatten() for k, v in model_w.items() if k in prunable_layers])
    k = int(len(all_values) * k)
    top_k_values, _ = torch.topk(all_values, k)
    threshold = top_k_values[-1]

    topk_weights = {}
    for key, tensor in model_w.items():
        if key in prunable_layers:
            topk_weights[key] = torch.where(tensor > threshold, tensor, torch.zeros(1, dtype=tensor.dtype, device=tensor.device))
        else:
            topk_weights[key] = tensor
    return topk_weights


def get_threshold_list(k_percent, local_model_tuple_list, prunable_layers):
    threshold_list = []
    for local_data, model_w in local_model_tuple_list:
        all_values = torch.cat([v.flatten() for k, v in model_w.items() if k in prunable_layers])
        k = int(len(all_values) * k_percent)
        top_k_values, _ = torch.topk(all_values, k)
        threshold_list.append(top_k_values[-1].item())
    return threshold_list






# --- Sparsity Tools for DisPFL ---
def set_masks(masks):
    masks = masks

def init_masks(params, sparsities):
    masks ={}
    for name in params:
        masks[name] = torch.zeros_like(params[name])
        dense_numel = int((1-sparsities[name])*torch.numel(masks[name]))
        if dense_numel > 0:
            temp = masks[name].view(-1)
            perm = torch.randperm(len(temp))
            perm = perm[:dense_numel]
            temp[perm] = 1
    return masks

def calculate_sparsities(params, args, logger, tabu=[], distribution="ERK", sparse=0.5):
    """
    Calculates per-layer sparsities using either a uniform or ERK distribution.
    This is a standalone utility function.
    """
    spasities = {}
    if distribution == "uniform":
        for name in params:
            if name not in tabu:
                spasities[name] = 1 - args.model.dense_ratio
            else:
                spasities[name] = 0
    elif distribution == "ERK":
        logger.info('Initializing by ERK')
        total_params = 0
        for name in params:
            total_params += params[name].numel()
        is_epsilon_valid = False
        dense_layers = set()
        density = sparse
        while not is_epsilon_valid:
            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name in params:
                if name in tabu:
                    dense_layers.add(name)
                n_param = np.prod(params[name].shape)
                n_zeros = n_param * (1 - density)
                n_ones = n_param * density
                if name in dense_layers:
                    rhs -= n_zeros
                else:
                    rhs += n_ones
                    raw_probabilities[name] = (np.sum(params[name].shape) / np.prod(params[name].shape)) ** args.model.erk_power_scale
                    divisor += raw_probabilities[name] * n_param
            
            epsilon = rhs / divisor if divisor != 0 else 0
            max_prob = np.max(list(raw_probabilities.values())) if raw_probabilities else 0
            max_prob_one = max_prob * epsilon
            
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True
    
        for name in params:
            if name in dense_layers:
                spasities[name] = 0
            else:
                spasities[name] = (1 - epsilon * raw_probabilities[name])
                
    return spasities
