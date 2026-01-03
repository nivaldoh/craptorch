import numpy as np
from typing import Optional

from craptorch.core.tensor import Tensor
from craptorch.core.losses import BinaryCrossEntropyLoss

def test_unit_binary_cross_entropy_loss():
    loss_fn = BinaryCrossEntropyLoss()

    perfect_preds = Tensor([0.9999, 0.0001, 0.9999, 0.0001])
    targets = Tensor([1.0, 0.0, 1.0, 0.0])
    perf_loss = loss_fn.forward(perfect_preds, targets)
    assert perf_loss.data < 0.01

    bad_preds = Tensor([0.0001, 0.9999, 0.0001, 0.9999])
    targets = Tensor([1.0, 0.0, 1.0, 0.0])
    bad_loss = loss_fn.forward(bad_preds, targets)
    assert bad_loss.data > 5.0

    uniform_preds = Tensor([0.5, 0.5, 0.5, 0.5])
    targets = Tensor([1.0, 0.0, 1.0, 0.0])
    unif_loss = loss_fn.forward(uniform_preds, targets)
    expected_unif_loss = -np.log(0.5)
    assert np.allclose(expected_unif_loss, unif_loss.data, atol=0.01)

    boundary_preds = Tensor([0.0, 1.0, 0.0, 1.0])
    targets = Tensor([0.0, 1.0, 1.0, 0.0])
    boundary_loss = loss_fn.forward(boundary_preds, targets)
    assert not np.isnan(boundary_loss.data)
    assert not np.isinf(boundary_loss.data)

if __name__ == "__main__":
    test_unit_binary_cross_entropy_loss()