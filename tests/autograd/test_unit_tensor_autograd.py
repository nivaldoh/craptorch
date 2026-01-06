from craptorch.core.tensor import Tensor
from craptorch.core.autograd import AddBackward, MulBackward, MatMulBackward

import numpy as np

def test_unit_tensor_autograd():
    x = Tensor([2.0], requires_grad=True)
    y = x * 3
    z = y + 1

    z.backward()
    # assert np.allclose(y.grad, [1.0])
    assert np.allclose(x.grad, [3.0])

if __name__ == "__main__":
    test_unit_tensor_autograd()