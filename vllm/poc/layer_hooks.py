"""Per-round layer hooks for structure breaking.

Applies transformations between transformer layers to break
the model's learned output structure.
"""
from typing import List, Optional

import torch

from .gpu_random import generate_householder_vector, apply_householder, _seed_from_string, _uniform


class LayerHouseholderHook:
    """Per-round Householder reflections applied between transformer layers.
    
    These hooks apply the same transform to all nonces in a round (determined
    by block_hash). Combined with per-nonce hidden state transforms, this
    provides strong structure breaking.
    
    Usage:
        # At round init
        hooks = LayerHouseholderHook(model, block_hash, device, hidden_size)
        
        # Run forward passes...
        
        # At round end
        hooks.detach()
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        block_hash: str,
        device: torch.device,
        hidden_size: int,
    ):
        self.hooks: List = []
        self.reflection_vectors: List[torch.Tensor] = []
        self.block_hash = block_hash
        self._setup(model, block_hash, device, hidden_size)
    
    def _find_layers(self, model: torch.nn.Module) -> List[torch.nn.Module]:
        """Find transformer layers in a model-agnostic way."""
        # Try common patterns for different model architectures
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # Llama, Qwen, Mistral style
            return list(model.model.layers)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # GPT-2 style
            return list(model.transformer.h)
        elif hasattr(model, 'layers'):
            # Direct layers attribute
            return list(model.layers)
        return []
    
    def _setup(
        self,
        model: torch.nn.Module,
        block_hash: str,
        device: torch.device,
        hidden_size: int,
    ):
        """Setup hooks on all transformer layers."""
        layers = self._find_layers(model)
        
        for i in range(len(layers)):
            seed_str = f"{block_hash}_layer_{i}_householder"
            v = generate_householder_vector(seed_str, hidden_size, device)
            self.reflection_vectors.append(v)
            
            hook = layers[i].register_forward_hook(self._create_hook(i))
            self.hooks.append(hook)
    
    def _create_hook(self, layer_idx: int):
        """Create a forward hook that applies Householder reflection.
        
        vLLM decoder layers typically return (hidden_states, residual).
        We must transform BOTH to prevent residual connections from
        preserving untransformed values.
        
        EXPERIMENT: Also normalize to unit sphere at each layer to break
        magnitude-based structure accumulation.
        """
        def hook(module, input, output):
            v = self.reflection_vectors[layer_idx]
            
            def normalize_and_transform(x):
                # First normalize to unit sphere (break magnitude structure)
                x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
                # Then apply Householder reflection
                return apply_householder(x_norm, v.to(x_norm.dtype))
            
            if isinstance(output, tuple):
                if len(output) >= 2:
                    # (hidden_states, residual, ...) format - transform both
                    hidden = output[0]
                    residual = output[1]
                    rest = output[2:] if len(output) > 2 else ()
                    transformed_hidden = normalize_and_transform(hidden)
                    transformed_residual = normalize_and_transform(residual)
                    return (transformed_hidden, transformed_residual) + rest
                else:
                    # Single element tuple
                    hidden = output[0]
                    transformed = normalize_and_transform(hidden)
                    return (transformed,)
            else:
                transformed = normalize_and_transform(output)
                return transformed
        
        return hook
    
    def detach(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.reflection_vectors = []
    
    @property
    def num_layers(self) -> int:
        """Number of layers with hooks attached."""
        return len(self.hooks)


def _generate_signs(seed_str: str, dim: int, device: torch.device) -> torch.Tensor:
    """Generate random ±1 signs for a layer."""
    seed = _seed_from_string(seed_str)
    u = _uniform(seed, dim, device)
    return (u > 0.5).float() * 2 - 1  # Convert to ±1


class LayerSignsHook:
    """Per-round random signs applied between transformer layers.
    
    More disruptive than Householder because it independently flips
    each dimension's sign, breaking correlations between dimensions.
    
    Usage:
        # At round init
        hooks = LayerSignsHook(model, block_hash, device, hidden_size)
        
        # Run forward passes...
        
        # At round end
        hooks.detach()
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        block_hash: str,
        device: torch.device,
        hidden_size: int,
    ):
        self.hooks: List = []
        self.sign_vectors: List[torch.Tensor] = []
        self.block_hash = block_hash
        self.call_counts: List[int] = []  # Debug: track hook calls
        self._setup(model, block_hash, device, hidden_size)
    
    def _find_layers(self, model: torch.nn.Module) -> List[torch.nn.Module]:
        """Find transformer layers in a model-agnostic way."""
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return list(model.model.layers)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return list(model.transformer.h)
        elif hasattr(model, 'layers'):
            return list(model.layers)
        return []
    
    def _setup(
        self,
        model: torch.nn.Module,
        block_hash: str,
        device: torch.device,
        hidden_size: int,
    ):
        """Setup hooks on all transformer layers."""
        layers = self._find_layers(model)
        
        for i in range(len(layers)):
            seed_str = f"{block_hash}_layer_{i}_signs"
            signs = _generate_signs(seed_str, hidden_size, device)
            self.sign_vectors.append(signs)
            
            hook = layers[i].register_forward_hook(self._create_hook(i))
            self.hooks.append(hook)
    
    def _create_hook(self, layer_idx: int):
        """Create a forward hook that applies random sign flips.
        
        vLLM decoder layers typically return (hidden_states, residual).
        We must transform BOTH to prevent residual connections from
        preserving untransformed values.
        """
        def hook(module, input, output):
            signs = self.sign_vectors[layer_idx]
            
            if isinstance(output, tuple):
                if len(output) >= 2:
                    hidden = output[0]
                    residual = output[1]
                    rest = output[2:] if len(output) > 2 else ()
                    # Element-wise multiply by ±1 signs
                    transformed_hidden = hidden * signs.to(hidden.dtype)
                    transformed_residual = residual * signs.to(residual.dtype)
                    return (transformed_hidden, transformed_residual) + rest
                else:
                    hidden = output[0]
                    transformed = hidden * signs.to(hidden.dtype)
                    return (transformed,)
            else:
                return output * signs.to(output.dtype)
        
        return hook
    
    def detach(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.sign_vectors = []
    
    @property
    def num_layers(self) -> int:
        """Number of layers with hooks attached."""
        return len(self.hooks)

