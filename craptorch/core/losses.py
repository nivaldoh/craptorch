import numpy as np
from typing import Optional

from craptorch.core.tensor import Tensor

EPSILON = 1e-7  # Small value to prevent log(0) and numerical instability

def log_softmax(x: Tensor, dim: int = -1):
    # trick to avoid overflow
    max_vals = np.max(x.data, axis=dim, keepdims=True)
    subtracted = x.data - max_vals
    log_sum_exp = np.log(np.sum(np.exp(subtracted), axis=dim, keepdims=True))
    return Tensor(x.data - max_vals - log_sum_exp)

class Loss():
    def forward(self, predictions: Tensor, targets: Tensor):
        raise NotImplementedError()

    def backward(self):
        raise NotImplementedError()

class MSELoss(Loss):
    def forward(self, predictions, targets):
        squared_diffs = (predictions.data - targets.data) ** 2
        mse = np.mean(squared_diffs)
        return Tensor(mse)

    def __call__(self, predictions, targets):
        return self.forward(predictions, targets)
    
    def __backward__(self):
        pass

class CrossEntropyLoss(Loss):
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        log_probs = log_softmax(logits, dim=-1)

        batch_size = logits.shape[0]
        target_indices = targets.data.astype(int)
        b = np.arange(batch_size)

        # for each batch, get the probability that was predicted for the class that should've had max prob with high confidence
        selected_log_probs = log_probs.data[b, target_indices]

        cross_entr = -np.mean(selected_log_probs)
        return Tensor(cross_entr)

    def __call__(self, logits, targets) -> Tensor:
        return self.forward(logits, targets)

    def backward(self) -> Tensor:
        pass