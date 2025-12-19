from craptorch.core.tensor import Tensor

import numpy as np

def test_unit_tensor_creation():
    """Test tensor creation with various data types"""
    print("🧪 Unit Test: Tensor Creation...")

    scalar = Tensor(1.0)
    assert scalar.data == 1.0
    assert scalar.shape == ()
    assert scalar.size == 1
    assert scalar.requires_grad == False
    assert scalar.grad is None
    assert scalar.dtype == np.float32

    v = Tensor([1, 2, 3])
    assert np.array_equal(v.data, np.array([1,2,3], dtype=np.float32))
    assert v.shape == (3,)
    assert v.size == 3

    m = Tensor([[1,2], [3,4]])
    assert np.array_equal(m.data, np.array([[1,2], [3,4]], dtype=np.float32))
    assert m.shape == (2, 2)
    assert m.size == 4

    grad_tensor = Tensor([1,2], requires_grad=True)
    assert grad_tensor.requires_grad == True
    assert grad_tensor.grad is None

    print("✅ Tensor creation works correctly!")

if __name__ == "__main__":
    test_unit_tensor_creation()