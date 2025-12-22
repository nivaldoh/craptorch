from craptorch.core.tensor import Tensor
from craptorch.core.activations import TOLERANCE, Softmax

import numpy as np

def test_unit_softmax():
    softmax = Softmax()

    x = Tensor([1,2,3])
    act = softmax.forward(x)
    assert np.allclose(np.sum(act.data), [1.0])
    assert np.all(act.data > 0), "softmax values > 0"
    assert np.all(act.data < 1), "softmax values < 1"

    max_input_idx = np.argmax(x.data)
    max_output_idx = np.argmax(act.data)
    assert max_input_idx == max_output_idx, "the largest value should have the largest prob"

    x = Tensor([1000, 1001, 1002])
    act = softmax.forward(x)
    assert np.allclose(np.sum(act.data), 1.0), "probs add up to 1"
    assert not np.any(np.isnan(act.data)), "has nan"
    assert not np.any(np.isinf(act.data)), "has inf"

    x = Tensor([[1,2], [3,4]])
    act = softmax.forward(x)
    assert act.shape == (2,2), "batch shape not kept"
    row_sums = np.sum(act.data, axis=-1)
    assert np.allclose(row_sums, [1.0, 1.0]), "sums should be over each element in batch"

if __name__ == "__main__":
    test_unit_softmax()