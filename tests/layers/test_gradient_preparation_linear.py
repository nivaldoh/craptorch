from craptorch.core.tensor import Tensor
from craptorch.core.activations import ReLU, Softmax
from craptorch.core.layers import Linear, XAVIER_SCALE_FACTOR

import numpy as np

def test_gradient_preparation_linear():
    layer = Linear(10,5)

    assert layer.weight.requires_grad == True
    assert layer.bias.requires_grad == True

    assert hasattr(layer.weight, 'grad')
    assert hasattr(layer.bias, 'grad')

    params = layer.parameters()
    assert len(params) == 2
    assert all(p.requires_grad for p in params)

if __name__ == "__main__":
    test_gradient_preparation_linear()