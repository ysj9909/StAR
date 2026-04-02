import torch
import torch.nn as nn
from contextlib import nullcontext
try:
    from torch.distributed.fsdp import FullyShardedDataParallel as _TorchFSDP
except ImportError:
    _TorchFSDP = None                         # 싱글 GPU 환경

def is_fsdp_wrapped(m: nn.Module) -> bool:
    """모듈이 FSDP 래퍼인지 속성으로 판단."""
    return hasattr(m, "_fsdp_wrapped_module")

def summon_full_params_if_needed(mods, recurse=False, writeback=False):
    """
    FSDP 모듈이 하나라도 있으면 gather, 아니면 no‑op context 반환.
    """
    if _TorchFSDP and any(is_fsdp_wrapped(m) for m in mods):
        return _TorchFSDP.summon_full_params(mods,
                                             recurse=recurse,
                                             writeback=writeback)
    return nullcontext()