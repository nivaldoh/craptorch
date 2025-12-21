from craptorch.core.tensor import Tensor
from craptorch.core.activations import TOLERANCE, ReLU

import numpy as np

def test_unit_relu():
    relu = ReLU()

    x = Tensor([-2,-1,0,1,2])
    res = relu.forward(x)
    expected = [0,0,0,1,2]
    assert np.allclose(res.data, expected), f"ReLU failed. Expected {expected}, got {res.data}"

    x = Tensor([-5, -3, -1])
    result = relu.forward(x)
    assert np.allclose(result.data, [0, 0, 0]), "ReLU should zero all negative values"

    x = Tensor([1, 3, 5])
    result = relu.forward(x)
    assert np.allclose(result.data, [1, 3, 5]), "ReLU should preserve all positive values"

    x = Tensor([-1, -2, -3, 1])
    res = relu.forward(x)
    zeros = np.sum(res.data == 0)
    assert zeros == 3, f"ReLu should create sparsity, got {zeros} zeroes out of 4"

if __name__ == "__main__":
    test_unit_relu()