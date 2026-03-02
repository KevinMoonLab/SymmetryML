# symdisc/vector_fields/images.py
from __future__ import annotations
from typing import Dict, Optional, Any, Tuple, Literal
import torch
import torch.nn.functional as F

PaddingMode = Literal["same", "valid"]
KernelMap = Dict[str, torch.Tensor]

# Global registry for pre-defined kernels (stored as 2D tensors, shape (H, W))
_KERNELS: KernelMap = {}

def register_kernel(name: str, kernel_2d: torch.Tensor) -> None:
    """
    Register a 2D kernel by name. Kernel should be shape (H, W), dtype float.
    It will be treated as a *single-channel* base kernel and expanded at call time.
    """
    if kernel_2d.ndim != 2:
        raise ValueError("Expected a 2D kernel of shape (H, W).")
    if not kernel_2d.is_floating_point():
        kernel_2d = kernel_2d.float()
    _KERNELS[name] = kernel_2d.detach().clone()

def get_kernel(name: str) -> torch.Tensor:
    if name not in _KERNELS:
        raise KeyError(f"Kernel '{name}' is not registered.")
    return _KERNELS[name]

def _prepare_conv_weight(
    base2d: torch.Tensor,
    channels: int,
    *,
    device,
    dtype,
    groups: Literal["depthwise", "full"] = "depthwise",
    normalize: bool = False,
) -> Tuple[torch.Tensor, int]:
    """
    Expand a base (H, W) kernel to a 4D conv weight matching groups strategy.
    - depthwise: (C, 1, H, W), groups=C
    - full     : (C, C, H, W), groups=1  (same kernel applied with cross-channel mixing)
    """
    k = base2d.to(device=device, dtype=dtype)
    if normalize:
        s = k.abs().sum()
        if s > 0:
            k = k / s

    if groups == "depthwise":
        weight = k.view(1, 1, *k.shape).repeat(channels, 1, 1, 1)  # (C,1,H,W)
        return weight, channels
    else:
        weight = k.view(1, 1, *k.shape).repeat(channels, channels, 1, 1)  # (C,C,H,W)
        return weight, 1

def conv2d_field_from_kernel(
    kernel_name: str,
    *,
    padding: PaddingMode = "same",
    groups: Literal["depthwise", "full"] = "depthwise",
    normalize_kernel: bool = False,
):
    """
    Returns a vector field X(img) = conv2d(img, K) with fixed kernel K.
    - img: (N, C, H, W)  ->  (N, C, H, W)
    - The kernel is looked up by name (pre-registered), then expanded to conv weight.
    """
    def X(img: torch.Tensor, *, meta=None) -> torch.Tensor:
        if img.ndim != 4:
            raise AssertionError("Expected NCHW image.")
        N, C, H, W = img.shape
        base2d = get_kernel(kernel_name)
        weight, g = _prepare_conv_weight(base2d, C, device=img.device, dtype=img.dtype,
                                         groups=groups, normalize=normalize_kernel)
        return F.conv2d(img, weight, bias=None, stride=1, padding=padding, groups=g)
    return X


def gaussian_blur_of_gradient_field(
    kernel_name: str,
    *,
    padding: PaddingMode = "same",
    groups: Literal["depthwise", "full"] = "depthwise",
    normalize_kernel: bool = False,
    require_grad: bool = True,
):
    """
    Vector field that acts on the *provided gradient* (e.g., model grad w.r.t input):
        X(img; grad) = conv2d(grad, K)

    - Expects `grad` with shape (N, C, H, W).
    - Does NOT compute any image gradient internally.
    """
    def X(img: torch.Tensor, *, meta: Optional[Dict[str, Any]] = None, grad: Optional[torch.Tensor] = None) -> torch.Tensor:
        if require_grad and grad is None:
            raise ValueError("gaussian_blur_of_gradient_field requires `grad` of shape (N,C,H,W).")
        g = grad if grad is not None else img  # fallback only if you explicitly set require_grad=False
        assert g.ndim == 4, "Expected gradient as NCHW."
        N, C, H, W = g.shape
        base2d = get_kernel(kernel_name)
        weight, grp = _prepare_conv_weight(base2d, C, device=g.device, dtype=g.dtype,
                                           groups=groups, normalize=normalize_kernel)
        return F.conv2d(g, weight, bias=None, stride=1, padding=padding, groups=grp)
    return X


def power_law_gamma_field(
    *,
    eps: float = 1e-3,
    enforce_domain: bool = True,
):
    """
    Vector field for gamma (power-law) adjustment on images.

    X(img) = img * log(img), applied per-channel, per-pixel, assuming img in (0,1).
    If enforce_domain=True, we clamp to [eps, 1-eps] before log to avoid NaNs.

    Parameters
    ----------
    eps : float
        Lower/upper clamp margins when enforce_domain=True.
    enforce_domain : bool
        If True, clamp to [eps, 1-eps] before log. If False, assumes input is already in (0,1).

    Returns
    -------
    callable
        X: (N, C, H, W) -> (N, C, H, W)
    """
    def X(img: torch.Tensor, *, meta: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        assert img.ndim == 4, "Expected NCHW image."
        x = img
        if enforce_domain:
            x = x.clamp(min=eps, max=1.0 - eps)
        return x * torch.log(x)
    return X

