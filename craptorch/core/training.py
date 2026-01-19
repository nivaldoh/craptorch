import numpy as np
import pickle
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import sys
import os

from craptorch.core.tensor import Tensor
from craptorch.core.layers import Linear
from craptorch.core.losses import MSELoss, CrossEntropyLoss
from craptorch.core.optimizers import SGD, AdamW

DEFAULT_MAX_LR = 0.1  # Default maximum learning rate for cosine schedule
DEFAULT_MIN_LR = 0.01  # Default minimum learning rate for cosine schedule
DEFAULT_TOTAL_EPOCHS = 100  # Default total epochs for learning rate schedule

class CosineSchedule:
    def __init__(self, max_lr: float = DEFAULT_MAX_LR, min_lr: float = DEFAULT_MIN_LR, total_epochs: int = DEFAULT_TOTAL_EPOCHS):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs

    def get_lr(self, epoch: int) -> float:
        if epoch >= self.total_epochs:
            return self.min_lr

        cosine_factor = (1 + np.cos(np.pi * epoch / self.total_epochs)) / 2
        return self.min_lr + (self.max_lr - self.min_lr) * cosine_factor

def clip_grad_norm(parameters: List, max_norm: float = 1.0) -> float:
    if not parameters:
        return 0.0
    
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            if isinstance(param.grad, np.ndarray):
                grad_data = param.grad
            else:
                grad_data = param.grad.data
            total_norm += np.sum(grad_data ** 2)

    total_norm = np.sqrt(total_norm)

    if total_norm > max_norm:
        clip_coef = max_norm/total_norm
        for param in parameters:
            if param.grad is not None:
                if isinstance(param.grad, np.ndarray):
                    param.grad = param.grad * clip_coef
                else:
                    param.grad.data = param.grad.data * clip_coef
    
    return float(total_norm)