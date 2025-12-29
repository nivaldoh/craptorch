from craptorch.core.tensor import Tensor
from craptorch.core.activations import ReLU, Softmax
from craptorch.core.layers import Dropout

import numpy as np

def test_unit_dropout_layer():
    dropout = Dropout(0.5)
    assert dropout.p == 0.5

    x = Tensor([1,2,3,4])
    y_inference = dropout.forward(x, training=False)
    assert np.array_equal(x.data, y_inference.data)

    dropout_zero = Dropout(0.0)
    zero_inference = dropout_zero.forward(x, training=True)
    assert np.array_equal(zero_inference.data, x.data)

    dropout_full = Dropout(1.0)
    full_inf = dropout_full.forward(x)
    assert np.allclose(full_inf.data, 0)

    np.random.seed(123)
    x_large = Tensor(np.ones((1000,)))
    y_train = dropout.forward(x_large, training=True)

    non_zero_count = np.count_nonzero(y_train.data)
    expected = 500
    std_err = np.sqrt(1000 * 0.5 * 0.5)
    lower_bound = expected - 3*std_err
    upper_bound = expected + 3*std_err
    assert lower_bound < non_zero_count < upper_bound

    surviving_values = y_train.data[y_train.data != 0]
    expected_val = 2.0  # 1.0 / (1 - 0.5)
    assert np.allclose(surviving_values, expected_val)

    params = dropout.parameters()
    assert len(params) == 0

    try:
        d = Dropout(-0.2)
        assert False
    except ValueError:
        pass

    try:
        d = Dropout(1.11)
        assert False
    except ValueError:
        pass

if __name__ == "__main__":
    test_unit_dropout_layer()