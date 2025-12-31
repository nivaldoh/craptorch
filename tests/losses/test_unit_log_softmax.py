import numpy as np
from typing import Optional

from craptorch.core.tensor import Tensor
from craptorch.core.losses import log_softmax

def test_unit_log_softmax():
    x = Tensor([[1.0, 2.0, 3.0], [0.1, 0.2, 0.9]])
    res = log_softmax(x, dim=-1)
    
    assert res.shape == x.shape

    softmax_res = np.exp(res.data)
    row_sum = np.sum(softmax_res, axis=-1)
    assert np.allclose(row_sum, 1.0, atol=1e-6)

    large_x = Tensor([99.0, 100.0, 100.1])
    large_res = log_softmax(large_x, dim=-1)
    assert not np.any(np.isnan(large_res.data))
    assert not np.any(np.isinf(large_res.data))

if __name__ == "__main__":
    test_unit_log_softmax()