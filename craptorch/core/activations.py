import numpy as np
from typing import Optional

from craptorch.core.tensor import Tensor

# Constants for numerical comparisons
TOLERANCE = 1e-10  # Small tolerance for floating-point comparisons in tests

# Export only activation classes
__all__ = ['Sigmoid', 'ReLU', 'Tanh', 'GELU', 'Softmax']

class Activation:
    def parameters(self):
        return []

    def forward(self, x: Tensor):
        pass
    
    def backward(self, grad: Tensor):
        pass

    def __call__(self, x: Tensor):
        return self.forward(x)

class Sigmoid (Activation):
    def forward(self, x: Tensor):
        """Sigmoid activation: sigma(x) = 1/(1 + e^(-x))"""
        # Clip extreme values to prevent overflow (sigmoid(-500) ~ 0, sigmoid(500) ~ 1)
        # Clipping at |500| ensures exp() stays within float64 range
        z = np.clip(-x.data, -500, 500)

        # Different calculation for pos/neg for numerical stability
        result_data = np.zeros_like(z)
        pos_mask = z >= 0
        result_data[pos_mask] = 1.0/(1.0 + np.exp(-z[pos_mask]))

        neg_mask = z < 0
        exp_z = np.exp(z[neg_mask])
        result_data[neg_mask] = exp_z / (1.0 + exp_z)

        return Tensor(result_data)

    def backward(self, grad: Tensor):
        pass