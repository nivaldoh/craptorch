from craptorch.core.tensor import Tensor

import numpy as np

def test_unit_matmul():
    print("🧪 Unit Test: Matrix Multiplication...")

    a = Tensor([[1,2], [3,4]])
    b = Tensor([[2,3], [3,2]])

    res = a.matmul(b)
    assert np.array_equal(res.data, np.array([[8,7], [18,17]], dtype=np.float32))

    c = Tensor([[1, 2, 3], [4, 5, 6]])  # 2×3 (like batch_size=2, features=3)
    d = Tensor([[7, 8], [9, 10], [11, 12]])  # 3×2 (like features=3, outputs=2)
    result = c.matmul(d)
    # Expected: [[1×7+2×9+3×11, 1×8+2×10+3×12], [4×7+5×9+6×11, 4×8+5×10+6×12]]
    expected = np.array([[58, 64], [139, 154]], dtype=np.float32)
    assert np.array_equal(result.data, expected)

    # Test matrix-vector multiplication (common in forward pass)
    matrix = Tensor([[1, 2, 3], [4, 5, 6]])  # 2×3
    vector = Tensor([1, 2, 3])  # 3×1 (conceptually)
    result = matrix.matmul(vector)
    # Expected: [1×1+2×2+3×3, 4×1+5×2+6×3] = [14, 32]
    expected = np.array([14, 32], dtype=np.float32)
    assert np.array_equal(result.data, expected)

    try:
        incompatible_a = Tensor([[1,2],[3,4]])
        incompatible_b = Tensor([[1,2,3],[4,5,6],[7,8,9]])
        print(incompatible_a.shape)
        print(incompatible_b.shape)
        res = incompatible_a.matmul(incompatible_b)
        assert False, "Failed to raise ValueError for incompatible shapes"
    except ValueError as e:
        assert "Inner dims must match" in str(e)
        assert "2 != 3" in str(e)

if __name__ == "__main__":
    test_unit_matmul()