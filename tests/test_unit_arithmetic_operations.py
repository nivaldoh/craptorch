from craptorch.core.tensor import Tensor

import numpy as np

def test_unit_arithmetic_operations():
    print("🧪 Unit Test: Arithmetic Operations...")

    a = Tensor([1,2,3])
    b = Tensor([4,5,6])

    res = a + b
    assert np.array_equal(res.data, np.array([5,7,9], dtype=np.float32))

    res = a + 10
    assert np.array_equal(res.data, np.array([11,12,13], dtype=np.float32))

    res = a + 1
    assert np.array_equal(res.data, np.array([2,3,4], dtype=np.float32))
    res = a * 10
    assert np.array_equal(res.data, np.array([10,20,30], dtype=np.float32))
    
    v = Tensor([1,2])
    m = Tensor([[3,4], [5,6]])
    res = v + m
    assert np.array_equal(res.data, np.array([[4,6], [6,8]], dtype=np.float32))
    res = m + v
    assert np.array_equal(res.data, np.array([[4,6], [6,8]], dtype=np.float32))
    n = Tensor([[1], [2]])
    res = n * m
    assert np.array_equal(res.data, np.array([[3,4], [10,12]], dtype=np.float32))

    res = b-a
    assert np.array_equal(res.data, np.array([3,3,3], dtype=np.float32))

    result = a * 2
    assert np.array_equal(result.data, np.array([2, 4, 6], dtype=np.float32))

    result = b / 2
    assert np.array_equal(result.data, np.array([2.0, 2.5, 3.0], dtype=np.float32))

    normalized = (a - 2) / 2
    expected = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    assert np.allclose(normalized.data, expected)

    print("✅ Arithmetic operations work correctly!")


if __name__ == "__main__":
    test_unit_arithmetic_operations()