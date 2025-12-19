from craptorch.core.tensor import Tensor

import numpy as np

def test_unit_shape_manipulation():
    print("🧪 Unit Test: tensor manipulation...")

    t = Tensor([1,2,3,4,5,6])
    reshaped = t.reshape(2,3)
    assert reshaped.shape == (2,3)
    expected = np.array([[1,2,3], [4,5,6]], dtype=np.float32)
    assert np.array_equal(reshaped.data, expected)

    reshaped2 = t.reshape((3,2))
    assert reshaped2.shape == (3,2)
    expected = np.array([[1,2], [3,4], [5,6]], dtype=np.float32)
    assert np.array_equal(expected, reshaped2.data)

    reshaped_unk = t.reshape(2,-1)
    assert reshaped_unk.shape == (2,3)
    expected = np.array([[1,2,3], [4,5,6]], dtype=np.float32)
    assert np.array_equal(reshaped_unk.data, expected)

    try:
        t.reshape(6,2)
        assert False, "Should have failed to reshape"
    except ValueError as e:
        assert "Total elements must match" in str(e)
        assert "6 != 12"

    m = Tensor([[1,2,3], [4,5,6,]])
    transposed = m.transpose()
    assert transposed.shape == (3,2)
    expected = np.array([[1,4], [2,5], [3,6]], dtype=np.float32)
    assert np.array_equal(transposed.data, expected)

    one_d = Tensor([1,2,3])
    one_d_transposed = one_d.transpose()
    assert np.array_equal(one_d_transposed.data, one_d.data)

    three_d = Tensor([[[1,2], [3,4]], [[5,6], [7,8]]])
    three_d_transp = three_d.transpose(0, 2)
    assert three_d_transp.shape == (2,2,2)

    print("✅ Tensor manipulation works correctly!")

if __name__ == "__main__":
    test_unit_shape_manipulationtest_unit_tensor_creation()