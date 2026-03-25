import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from collections import OrderedDict

class SparseModel(nn.Module):
    """
    A wrapper class for a PyTorch model to handle pruning logic internally.

    This class encapsulates the application of masks and the conversion between
    pruned and unpruned state dictionaries, presenting a clean interface to
    the rest of the application.
    """
    def __init__(self, model):
        super().__init__()
        # This assignment registers 'model' as a submodule, storing it in self._modules.
        self.add_module('model', model) # self.model = model
        self.pruned_layers = set()

    def forward(self, x):
        """Performs a forward pass through the underlying model."""
        return self._modules['model'].forward(x)


    def apply_masks(self, masks):
        self.remove_pruning() # Ensure model is in a clean state first
        # Access the underlying model directly
        for name, module in self._modules['model'].named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                param_name = f"{name}.weight"
                if param_name in masks and masks[param_name] is not None:
                    # Get the device from the model layer itself (e.g., 'cuda:0')
                    device = module.weight.device
                    # Explicitly move the mask to that same device inside the client thread
                    mask_on_device = masks[param_name].to(device)
                    
                    # Now, use the device-correct mask for pruning
                    prune.custom_from_mask(module, name='weight', mask=mask_on_device)
                    
                    self.pruned_layers.add(name)

    def remove_pruning(self):
        """Removes all pruning re-parametrizations from the model."""
        # Access the underlying model directly
        for name, module in self._modules['model'].named_modules():
            if name in self.pruned_layers:
                 if prune.is_pruned(module):
                    prune.remove(module, 'weight')
        self.pruned_layers.clear()

    def state_dict(self, *args, **kwargs):
        """
        Overrides the default state_dict() method. It returns a "flattened"
        state dictionary by computing the final weights from the masks.
        """
        kwargs.pop('destination', None)
        # Access the underlying model directly to avoid recursion.
        pruned_state_dict = self._modules['model'].state_dict(*args, **kwargs)
        flat_state_dict = OrderedDict()

        for key, value in pruned_state_dict.items():
            if key.endswith(".weight_mask"):
                continue
            if key.endswith(".weight_orig"):
                base_name = key.rsplit('.', 1)[0]
                mask = pruned_state_dict[f"{base_name}.weight_mask"]
                flat_state_dict[f"{base_name}.weight"] = value * mask
            else:
                flat_state_dict[key] = value
        return flat_state_dict

    def load_state_dict(self, state_dict, strict=True):
        """
        Overrides the default load_state_dict() method. It loads a standard
        "flattened" state_dict into the model's internal pruned structure.
        """
        pruned_load_dict = OrderedDict()
        for key, value in state_dict.items():
            base_name = key.rsplit('.', 1)[0]
            if key.endswith(".weight") and base_name in self.pruned_layers:
                pruned_load_dict[f"{base_name}.weight_orig"] = value
            else:
                pruned_load_dict[key] = value
        
        # Must load non-strictly and access the underlying model directly.
        return self._modules['model'].load_state_dict(pruned_load_dict, strict=False)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """Override to delegate directly to underlying model without 'model.' prefix."""
        return self._modules['model'].named_parameters(prefix=prefix, recurse=recurse)

    def named_modules(self, memo=None, prefix: str = '', remove_duplicate: bool = True):
        """Override to delegate directly to underlying model without 'model.' prefix."""
        return self._modules['model'].named_modules(memo=memo, prefix=prefix, remove_duplicate=remove_duplicate)