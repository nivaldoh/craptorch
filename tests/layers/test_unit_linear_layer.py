from craptorch.core.tensor import Tensor
from craptorch.core.activations import ReLU, Softmax
from craptorch.core.layers import Linear, XAVIER_SCALE_FACTOR

import numpy as np

def test_unit_linear_layer():
    layer = Linear(784, 256)
    assert layer.in_features == 784
    assert layer.out_features == 256
    assert layer.weight.shape == (784, 256)
    assert layer.bias.shape == (256,)
    assert layer.weight.requires_grad == True
    assert layer.bias.requires_grad == True

    weight_std = np.std(layer.weight.data)
    expected_std = np.sqrt(XAVIER_SCALE_FACTOR/784)
    assert 0.5*expected_std < weight_std < 2.0*expected_std

    assert np.allclose(layer.bias.data, 0)

    x = Tensor(np.random.randn(32, 784))
    y = layer.forward(x)
    assert y.shape == (32, 256)

    layer_nobias = Linear(10, 5, bias=False)
    assert layer_nobias.bias is None
    params = layer_nobias.parameters()
    assert len(params) == 1

    params = layer.parameters()
    assert len(params) == 2
    assert params[0] is layer.weight
    assert params[1] is layer.bias

if __name__ == "__main__":
    test_unit_linear_layer()