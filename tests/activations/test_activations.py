from craptorch.core.tensor import Tensor
from craptorch.core.activations import TOLERANCE, Sigmoid, ReLU, GELU, Softmax

import numpy as np

def test_activations():
    softmax = Softmax()

    test_data = Tensor([[1,-1], [2,-2]])
    activations = [Sigmoid(), ReLU(), GELU(), Softmax()]

    for a in activations:
        act = a.forward(test_data)
        assert act.shape == (2,2)
        assert isinstance(act, Tensor)

    data_3d = Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])  # (2, 2, 3)
    softmax = Softmax()

    act_last = softmax(data_3d, dim=-1)
    assert act_last.shape == (2,2,3)
    last_dim_sums = np.sum(act_last.data, axis=-1)
    assert np.allclose(last_dim_sums, 1.0)

    x = Tensor([[-1,0,1,2]])
    relu = ReLU()
    hidden = relu.forward(x)
    softmax = Softmax()
    probs = softmax.forward(hidden)
    assert hidden.data[0,0] == 0
    assert np.allclose(np.sum(probs.data), 1.0)


if __name__ == "__main__":
    test_activations()