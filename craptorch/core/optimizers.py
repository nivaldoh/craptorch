import numpy as np
from typing import List, Union, Optional, Dict, Any

from craptorch.core.tensor import Tensor

DEFAULT_LEARNING_RATE_SGD = 0.01  # Default learning rate for SGD
DEFAULT_LEARNING_RATE_ADAM = 0.001  # Default learning rate for Adam/AdamW
DEFAULT_MOMENTUM = 0.9  # Default momentum for SGD
DEFAULT_BETA1 = 0.9  # First moment decay rate for Adam
DEFAULT_BETA2 = 0.999  # Second moment decay rate for Adam
DEFAULT_EPS = 1e-8  # Small epsilon for numerical stability in Adam
DEFAULT_WEIGHT_DECAY_ADAMW = 0.01  # Default weight decay for AdamW

class Optimizer:
    def __init__(self, params: List[Tensor]):
        if not isinstance(params, list):
            params = list(params)

        for i,p in enumerate(params):
            if not p.requires_grad:
                raise ValueError(f'Param {i} is not trainable')

        self.params = params
        self.step_count = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError('subclasses must implement step')