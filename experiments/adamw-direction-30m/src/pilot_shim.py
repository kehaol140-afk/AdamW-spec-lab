"""pilot_shim — verbatim V16 loss_sum_count (pilot_core.py:531-540), vendored to avoid
importing the whole V16 pilot package. Math byte-identical."""
import torch
import torch.nn.functional as F


def loss_sum_count(logits: torch.Tensor, ids: torch.Tensor, lengths: torch.Tensor):
    """One label shift; position mask uses lengths, never token values."""
    seq = ids.shape[1]
    target = ids[:, 1:]
    if logits.shape[1] not in (seq, seq - 1):
        raise ValueError(f"logits length={logits.shape[1]} incompatible with token window={seq}")
    mask = torch.arange(seq - 1, device=ids.device)[None, :] < (lengths[:, None] - 1)
    per = F.cross_entropy(logits[:, :target.shape[1]].float().reshape(-1, logits.shape[-1]),
                          target.reshape(-1), reduction="none").view_as(target)
    return (per * mask).sum(), int(mask.sum().item())
