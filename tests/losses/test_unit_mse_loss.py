import numpy as np
from typing import Optional

from craptorch.core.tensor import Tensor
from craptorch.core.losses import MSELoss, EPSILON

def test_unit_mse_loss():
    loss_fn = MSELoss()

    labels = Tensor([1.0, 2.0, 3.0])
    targets = Tensor([1.0, 2.0, 3.0])
    zero_loss = loss_fn.forward(labels, targets)
    assert np.allclose(zero_loss.data, 0.0, atol=EPSILON)

    labels = Tensor([1.0, 2.0, 3.0])
    targets = Tensor([1.5, 2.5, 2.8])
    l = loss_fn.forward(labels, targets)
    expected_loss = (0.25 + 0.25 + 0.04)/3
    assert np.allclose(l.data, expected_loss, atol=1e-6)

    random_pred = Tensor(np.random.randn(10))
    random_target = Tensor(np.random.randn(10))
    random_loss = loss_fn.forward(random_pred, random_target)
    assert random_loss.data >= 0

if __name__ == "__main__":
    test_unit_mse_loss()