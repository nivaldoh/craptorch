from craptorch.core.tensor import Tensor
from craptorch.core.autograd import AddBackward, MulBackward, MatMulBackward

import numpy as np

def test_unit_function_classes():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([4,5,6], requires_grad=True)
    add_func = AddBackward(a,b)
    grad_out = np.array([1,1,1])
    grad_a, grad_b = add_func.apply(grad_out)
    assert np.allclose(grad_a, grad_out)
    assert np.allclose(grad_b, grad_out)

    mul_func = MulBackward(a,b)
    grad_a, grad_b = mul_func.apply(grad_out)
    assert np.allclose(grad_a, b.data)
    assert np.allclose(grad_b, a.data)

    a_mat = Tensor([[1,2], [3,4]], requires_grad=True)
    b_mat = Tensor([[5,6], [7,8]], requires_grad=True)
    matmul_func = MatMulBackward(a_mat, b_mat)
    grad_out = np.ones((2,2))
    grad_a, grad_b = matmul_func.apply(grad_out)
    assert grad_a.shape == a_mat.shape
    assert grad_b.shape == b_mat.shape

if __name__ == "__main__":
    test_unit_function_classes()