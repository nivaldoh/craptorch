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

class SGD(Optimizer):
    def __init__(self, params: List[Tensor], lr: float = DEFAULT_LEARNING_RATE_SGD, momentum : float = 0.0, weight_decay: float = 0.0):
        super().__init__(params)

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Initialize momentum buffers (created lazily)
        self.momentum_buffers = [None for _ in self.params]

    def has_momentum(self) -> bool:
        return self.momentum > 0

    def get_momentum_state(self) -> Optional[List]:
        """Get momentum buffers for checkpointing."""
        if not self.has_momentum():
            return None
        return [buf.copy() if buf is not None else None
            for buf in self.momentum_buffers]

    def set_momentum_state(self, state:Optional[List]) -> None:
        """Restore momentum buffers for checkpointing."""
        if state is None or not self.has_momentum():
            return

        if len(state) != (self.momentum_buffers):
            raise ValueError('State len doesnt match optimizer params')

        for i, buf in enumerate(state):
            if buf is not None:
                self.momentum_buffers[i] = buf.copy()

    def step(self):
        for i,param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad
            if isinstance(grad, Tensor):
                grad_data = grad.data
            else:
                grad_data = grad

            if self.weight_decay != 0:
                grad_data = grad_data + self.weight_decay * param.data

            if self.momentum != 0:
                if self.momentum_buffers[i] is None:
                    self.momentum_buffers[i] = np.zeros_like(param.data)
                
                self.momentum_buffers[i] = self.momentum * self.momentum_buffers[i] * grad.data
                grad_data = self.momentum_buffers[i]

            param.data = param.data - self.lr * grad_data

        self.step_count += 1
