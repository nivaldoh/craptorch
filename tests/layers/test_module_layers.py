from craptorch.core.tensor import Tensor
from craptorch.core.activations import ReLU, Softmax
from craptorch.core.layers import Linear, Dropout, ReLU

import numpy as np

def test_module_layers():
    ReLU_class = ReLU

    layer1 = Linear(784, 128)
    activation1 = ReLU_class()
    dropout1 = Dropout(0.5)
    layer2 = Linear(128, 64)
    activation2 = ReLU_class()
    dropout2 = Dropout(0.3)
    layer3 = Linear(64, 10)

    batch_size = 16
    x = Tensor(np.random.randn(batch_size, 784))

    x = layer1.forward(x)
    x = activation1.forward(x)
    x = dropout1.forward(x)
    x = layer2.forward(x)
    x = activation2.forward(x)
    x = dropout2.forward(x)
    out = layer3.forward(x)

    assert out.shape == (batch_size, 10)

    all_params = layer1.parameters() + layer2.parameters() + layer3.parameters()
    expected_params = 6
    assert len(all_params) == expected_params

    for p in all_params:
        assert p.requires_grad == True

    test_x = Tensor(np.random.randn(4, 784))
    dropout_test = Dropout(0.5)
    train_out = dropout_test.forward(test_x, training=True)
    inference_out = dropout_test.forward(test_x, training=False)
    assert np.array_equal(test_x.data, inference_out.data)

if __name__ == "__main__":
    test_module_layers()