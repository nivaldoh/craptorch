import numpy as np
from typing import Optional

from craptorch.core.tensor import Tensor
from craptorch.core.losses import CrossEntropyLoss

def test_unit_cross_entropy_loss():
    loss_fn = CrossEntropyLoss()

    perfect_logits = Tensor([[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]])
    targets = Tensor([0,1])
    perfect_loss = loss_fn.forward(perfect_logits, targets)
    assert perfect_loss.data < 0.01

    uniform_logits = Tensor([[2.0,2.0,2.0], [0.5,0.5,0.5]])
    targets = Tensor([0,1])
    uniform_loss = loss_fn.forward(uniform_logits, targets)
    expected_unif_loss = np.log(3)
    assert np.allclose(expected_unif_loss, uniform_loss.data, atol=0.1)

    wrong_logits = Tensor([[10.0, -10.0, -10.0], [-10.0, -10.0, 10.0]])
    wrong_targets = Tensor([1,1])
    wrong_loss = loss_fn.forward(wrong_logits, wrong_targets)
    assert wrong_loss.data > 5.0

    large_logits = Tensor([[100.0, 50.0, 25.0]])
    targets = Tensor([0])
    large_loss = loss_fn.forward(large_logits, targets)
    assert not np.isnan(large_loss.data)
    assert not np.isinf(large_loss.data)

if __name__ == "__main__":
    test_unit_cross_entropy_loss()