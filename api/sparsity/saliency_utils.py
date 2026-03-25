# In api/saliency/saliency_utils.py
import torch


def get_mean_saliency_scores(scores_gathered):
    """
    Averages a list of saliency score dictionaries element-wise.
    This function is generic and works for any saliency metric.

    Args:
        scores_gathered (list): A list of score dictionaries, where each dict
                                maps layer names to score tensors.

    Returns:
        dict: A single dictionary with the averaged scores.
    """
    if not scores_gathered:
        return {}
    
    # Initialize the average dictionary with clones of the first entry
    avg_scores = {k: v.clone().detach() for k, v in scores_gathered[0].items()}
    
    # Sum the scores from the remaining entries
    for score_dict in scores_gathered[1:]:
        for k, v in score_dict.items():
            if k in avg_scores:
                avg_scores[k] += v
            else:
                # This case handles potential inconsistencies in layer naming
                avg_scores[k] = v.clone().detach()

    # Divide by the total number of entries to get the mean
    num_entries = len(scores_gathered)
    for k in avg_scores:
        avg_scores[k] /= num_entries
        
    return avg_scores


def create_mask_from_scores(scores_dict, keep_ratio, device):
    """
    Creates a binary mask by keeping a specified ratio of the top-scoring weights.
    This is a generic function for global magnitude pruning.

    Args:
        scores_dict (dict): A dictionary mapping layer names to their score tensors.
        keep_ratio (float): The fraction of weights to keep (e.g., 0.2 for 20% density).
        device: The torch device to place the final masks on.

    Returns:
        tuple: A tuple containing:
            - final_weight_mask (dict): A dictionary mapping parameter names to binary masks.
            - layer_wise_density (dict): A dictionary mapping layer names to their resulting density.
    """
    # Raise a specific error if no saliency scores were generated ---
    if not scores_dict:
        raise ValueError(
            "The saliency score dictionary is empty. This likely means that no prunable layers "
            "were matched with their gradients during the saliency calculation. Please check "
            "that the model architecture and parameter names are consistent in `model_trainer.py`."
        )
    
    # Flatten all scores from all layers into a single tensor
    all_scores = torch.cat([v.flatten() for v in scores_dict.values()])
    
    # Calculate the number of weights to keep
    num_params_to_keep = int(len(all_scores) * keep_ratio)
    
    if num_params_to_keep < 1:
        # Handle the edge case of keeping 0 parameters
        threshold = float('inf')
    else:
        # Find the score of the k-th largest element, which will be our threshold
        threshold = torch.topk(all_scores, num_params_to_keep, sorted=True).values[-1]

    final_weight_mask = {}
    layer_wise_density = {}
    with torch.no_grad():
        for name, scores in scores_dict.items():
            # Create a binary mask for each layer based on the global threshold
            mask = (scores >= threshold).float().to(device)
            
            # The mask name should match the parameter name in the model's state_dict
            param_name = f"{name}.weight"
            final_weight_mask[param_name] = mask
            layer_wise_density[param_name] = mask.mean().item()

    return final_weight_mask, layer_wise_density