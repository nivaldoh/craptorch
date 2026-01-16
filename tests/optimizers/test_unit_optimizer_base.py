import numpy as np

from craptorch.core.tensor import Tensor
from craptorch.core.optimizers import Optimizer

def test_unit_optimizer_base():
    p1 = Tensor([1.0, 2.0], requires_grad=True)
    p2 = Tensor([[3.0, 4.0], [5.0, 6.0]], requires_grad=True)

    p1.grad = Tensor([0.1, 0.2])
    p2.grad = Tensor([[0.3, 0.4], [0.5, 0.6]])

    opt = Optimizer([p1, p2])

    assert len(opt.params) == 2
    assert opt.params[0] is p1
    assert opt.params[1] is p2
    assert opt.step_count == 0

    opt.zero_grad()
    assert p1.grad is None
    assert p2.grad is None

    try:
        no_grad = Tensor([2.0], requires_grad=False)
        opt = Optimizer(no_grad)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "not trainable" in str(e)

if __name__ == "__main__":
    test_unit_optimizer_base()