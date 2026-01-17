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