from craptorch.core.tensor import Tensor
from craptorch.core.autograd import AddBackward, MulBackward, MatMulBackward

import numpy as np

def test_unit_tensor_autograd():
    x = Tensor([2.0], requires_grad=True)
    y = x * 3
    z = y + 1
    z.backward()
    assert np.allclose(y.grad, [1.0])
    assert np.allclose(x.grad, [3.0])

    # test matmul grads
    a = Tensor([[1.0, 2.0]], requires_grad=True)  # 1x2
    b = Tensor([[3.0], [4.0]], requires_grad=True)  # 2x1
    c = a.matmul(b)  # 1x1, result = [[11.0]]
    c.backward()
    assert np.allclose(a.grad, [[3.0, 4.0]]), f"Expected [[3.0, 4.0]], got {a.grad}"
    assert np.allclose(b.grad, [[1.0], [2.0]]), f"Expected [[1.0], [2.0]], got {b.grad}"

    # test multiple ops
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * 2      # y = [2, 4]
    z = y.sum()    # z = 6
    z.backward()
    assert np.allclose(y.grad, [1.0, 1.0])
    assert np.allclose(x.grad, [2.0, 2.0]), f"Expected [2.0, 2.0], got {x.grad}"

    # TODO: more autograd tests


if __name__ == "__main__":
    test_unit_tensor_autograd()