from craptorch.core.tensor import Tensor
from craptorch.core.autograd import AddBackward, MulBackward, MatMulBackward

import numpy as np

def test_module_autograd():
    # test multiple layers
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    W1 = Tensor([[0.5, 0.3, 0.1], [0.2, 0.4, 0.6]], requires_grad=True)
    b1 = Tensor([[0.1, 0.2, 0.3]], requires_grad=True)

    h1 = x.matmul(W1) + b1
    assert h1.shape == (1,3)
    assert h1.requires_grad == True

    W2 = Tensor([[0.1], [0.2], [0.3]], requires_grad=True)
    h2 = h1.matmul(W2)
    assert h2.shape == (1,1)

    simple_loss = h2 * h2
    simple_loss.backward()

    assert x.grad is not None
    assert W1.grad is not None
    assert b1.grad is not None
    assert W2.grad is not None
    assert x.grad.shape == x.shape
    assert W1.grad.shape == W1.shape

    # test gradient accumulation
    x = Tensor([2.0], requires_grad=True)

    y1 = x * 3
    y1.backward()
    first_grad = x.grad.copy()

    y2 = x * 5
    y2.backward()

    assert np.allclose(x.grad, first_grad + 5.0)

    # test complex ops
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[2.0, 1.0], [1.0, 2.0]], requires_grad=True)

    # ((a @ b) + a) * b
    temp1 = a.matmul(b)  # Matrix multiplication
    temp2 = temp1 + a    # Addition
    result = temp2 * b   # Element-wise multiplication
    final = result.sum() # Sum reduction

    final.backward()

    assert a.grad is not None
    assert b.grad is not None
    assert a.grad.shape == a.shape
    assert b.grad.shape == b.shape

if __name__ == "__main__":
    test_module_autograd()