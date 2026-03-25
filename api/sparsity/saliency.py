# SSFL saliency calculation methods

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import types
import torch.autograd as autograd

# ===================================================================================
# Helper functions for the auxiliary variable SSFL method (ssfl_aux)
# ===================================================================================
def ssfl_aux_forward_conv2d(self, x):
    return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                    self.stride, self.padding, self.dilation, self.groups)

def ssfl_aux_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)
# ===================================================================================


def calculate_ssfl_scores(model_trainer, batch, logger):
    """
    Calculates SSFL saliency scores using the |gradient * weight| formulation.
    This method is more direct than the auxiliary variable approach.

    Returns:
        dict: A dictionary mapping layer names to their saliency score tensors.
    """
    
    logger.debug("Calculating SSFL saliency scores...")
    # Ensure the model is on the correct device for gradient calculation
    device = next(model_trainer.model.parameters()).device
    
    # 1. Get gradients for a single batch
    logger.debug("    Calculating gradients for one batch ...")
    grads_dict = model_trainer.screen_gradients(batch, device)
    logger.debug("    ...Gradient calculation complete.")
    
    # 2. Get current model weights
    weights_dict = model_trainer.get_model_params()
    
    # DEBUG: Print the names to see what's available
    logger.debug(f"Gradient dict keys: {list(grads_dict.keys())[:5]}...")  # Show first 5
    logger.debug(f"Weights dict keys: {list(weights_dict.keys())[:5]}...")  # Show first 5
    logger.debug(f"Prunable parameter names: {model_trainer.prunable_parameter_names[:5]}...")  # Show first 5
    
    saliency_scores = {}
    with torch.no_grad():
        for name, weight in weights_dict.items():
            if name in grads_dict and name in model_trainer.prunable_parameter_names:
                # 3. Calculate saliency: |gradient * weight|
                saliency_scores[name.replace(".weight", "")] = torch.abs(grads_dict[name].cpu() * weight)
                logger.debug(f"Added saliency score for: {name}")
    
    logger.debug(f"Final saliency_scores keys: {list(saliency_scores.keys())}")
    return saliency_scores

# ===================================================================================
# SSFL method using auxiliary variables (weight_mask)
# ===================================================================================
def calculate_ssfl_aux_scores(model_trainer, batch, logger):
    """
    Calculates SSFL saliency scores using the auxiliary variable method.
    This involves monkey-patching the model's forward pass.
    NOTE: This method implicitly runs in `train` mode.
    """
    logger.debug("Calculating SSFL-aux saliency scores...")
    
    model = model_trainer.model
    device = next(iter(model.parameters())).device
    
    # Create a fresh copy of the model
    cp_model = copy.deepcopy(model)
    # NOTE: cp_model.eval() is NOT called, to precisely mimic the old implementation.

    # Monkey-patch the Linear and Conv2d layers
    for layer in cp_model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight)).to(device)
            layer.weight.requires_grad = False

        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(ssfl_aux_forward_conv2d, layer)

        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(ssfl_aux_forward_linear, layer)

    cp_model.to(device)
    
    # Compute gradients
    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)
    cp_model.zero_grad()
    outputs = cp_model.forward(inputs)
    loss = F.nll_loss(outputs, targets)
    loss.backward()
    
    # Extract scores and format as a dictionary for API consistency
    saliency_scores = {}
    for name, layer in cp_model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            if hasattr(layer, 'weight_mask') and layer.weight_mask.grad is not None:
                saliency_scores[name] = torch.abs(layer.weight_mask.grad)

    del cp_model
    return saliency_scores
# ===================================================================================



# Registry of available saliency metrics
SALIENCY_METRICS = {
    'ssfl': calculate_ssfl_scores,
    'ssfl_aux': calculate_ssfl_aux_scores,
}


def get_saliency_scores(metric_name, model_trainer, batch, logger):
    """Add better error handling and validation"""
    if metric_name not in SALIENCY_METRICS:
        available = ', '.join(SALIENCY_METRICS.keys())
        raise ValueError(f"Unsupported saliency metric: '{metric_name}'. Available: {available}")
    
    try:
        score_func = SALIENCY_METRICS[metric_name]
        scores = score_func(model_trainer, batch, logger)
        
        if not scores:
            logger.warning(f"Saliency calculation for '{metric_name}' returned empty scores")
            
        return scores
    except Exception as e:
        logger.error(f"Failed to calculate {metric_name} scores: {e}")
        raise