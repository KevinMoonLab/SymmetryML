# symdisc/enforcement/regularization/schedules.py
from __future__ import annotations
from typing import Callable
import math

def constant(val: float) -> Callable[[int], float]:
    return lambda step: float(val)

def linear_warmup(max_val: float, warmup_steps: int) -> Callable[[int], float]:
    def fn(step: int) -> float:
        if step <= 0: return 0.0
        if step >= warmup_steps: return float(max_val)
        return float(max_val) * step / warmup_steps
    return fn

def cosine(max_val: float, total_steps: int) -> Callable[[int], float]:
    def fn(step: int) -> float:
        s = max(0, min(step, total_steps))
        return float(max_val) * 0.5 * (1 - math.cos(math.pi * s / total_steps))
    return fn

def exponential(max_val: float, tau: float) -> Callable[[int], float]:
    def fn(step: int) -> float:
        return float(max_val) * (1.0 - math.exp(-step / max(1.0, tau)))
    return fn

def jump(max_val: float, delay_steps: int) -> Callable[[int], float]:
    """
    lambda(step) = 0 for step < delay_steps; = max_val for step >= delay_steps
    """
    def fn(step: int) -> float:
        return 0.0 if step < delay_steps else float(max_val)
    return fn
