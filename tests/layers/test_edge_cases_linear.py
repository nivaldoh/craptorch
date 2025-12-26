from craptorch.core.tensor import Tensor
from craptorch.core.activations import ReLU, Softmax
from craptorch.core.layers import Linear, XAVIER_SCALE_FACTOR

import numpy as np

def test_edge_cases_layer():
    layer = Linear(10, 5)

    x_2d = Tensor(np.random.randn(1,10))
    y = layer.forward(x_2d)
    assert y.shape == (1, 5)

    x_empty = Tensor(np.random.randn(0,10))
    y_empty = layer.forward(x_empty)
    assert y_empty.shape == (0,5)

    layer_large = Linear(10,5)
    layer_large.weight.data = np.ones((10,5)) * 100
    x = Tensor(np.ones((1,10)))
    y = layer_large.forward(x)
    assert not np.any(np.isnan(y.data))
    assert not np.any(np.isinf(y.data))

    layer_nobias = Linear(10, 5, bias=False)
    x = Tensor(np.random.randn(4,10))
    y = layer_nobias.forward(x)
    assert y.shape == (4,5)

if __name__ == "__main__":
    test_edge_cases_layer()