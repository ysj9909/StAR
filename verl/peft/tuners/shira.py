import torch
from torch import nn

from peft.utils.integrations import dequantize_module_weight, gather_params_ctx
from peft.utils.other import transpose

try:
    from torch.distributed.fsdp.flat_param import FlatParameter
except ImportError:      # single‑GPU or no‑FSDP case
    FlatParameter = nn.Parameter
    
from .helpers_fsdp import is_fsdp_wrapped, summon_full_params_if_needed


def compute_sparsity_mask(weight: torch.Tensor, sparsity: float, multiplier: float = 0.0) -> torch.Tensor:
    """
    Return a mask that is 1 for the smallest-magnitude fraction of weights
    (determined by `sparsity`), and `multiplier` for the rest.
    """
    if sparsity >= 1.0:
        return torch.ones_like(weight)
    if sparsity <= 0.0:
        return torch.zeros_like(weight)
    
    flat = weight.abs().view(-1)
    k = int(sparsity * flat.numel())
    if k <= 0:
        return torch.zeros_like(weight)
    if k >= flat.numel():
        return torch.ones_like(weight)
    
    threshold, _ = torch.topk(flat, k, largest=False)
    thresh = threshold.max()
    return torch.where(
        weight.abs() <= thresh,
        torch.ones_like(weight),
        torch.full_like(weight, multiplier)
    )

def gather_weight(base_layer: nn.Module, fan_in_fan_out: bool) -> torch.Tensor:
    with gather_params_ctx(base_layer.parameters()):
        weight = dequantize_module_weight(base_layer)
    return transpose(weight, fan_in_fan_out)

# def compute_lora_weight(lora_A: nn.Module, lora_B: nn.Module) -> torch.Tensor:
#     """Compute ``B @ A`` in a way that works with FSDP wrappers."""

#     module_a = lora_A.module if hasattr(lora_A, "module") else lora_A
#     param = next(module_a.parameters())
#     dtype = param.dtype
#     device = param.device
#     eye = torch.eye(module_a.in_features, device=device, dtype=dtype)
#     with torch.no_grad():
#         return lora_B(lora_A(eye)).T


def _unflat_weight(linear: nn.Linear) -> torch.Tensor:
    """Return the 2‑D weight tensor, even if stored as FlatParameter."""
    W = linear.weight          # may be FlatParameter
    if isinstance(W, FlatParameter):
        return W.view(linear.out_features, linear.in_features)
    return W


def compute_lora_weight(lora_A: nn.Module, lora_B: nn.Module) -> torch.Tensor:
    """
    Differentiable B @ A that works for plain modules and FSDP‑wrapped ones.
    No torch.no_grad() ⇒ gradients flow back to both LoRA matrices.
    """
    if is_fsdp_wrapped(lora_A):
        fsdp_cls = lora_A.__class__
        with fsdp_cls.summon_full_params([lora_A, lora_B],
                                         recurse=False, writeback=False):
            Wa = _unflat_weight(lora_A)
            Wb = _unflat_weight(lora_B)
            return Wb @ Wa          
    Wa = _unflat_weight(lora_A)
    Wb = _unflat_weight(lora_B)
    return Wb @ Wa       