from craptorch.core.tensor import Tensor

import numpy as np

def test_unit_reduction_operations():
    print("🧪 Unit Test: reduction operations...")

    m = Tensor([[1,2,3], [4,5,6]])

    total = m.sum()
    assert total.data == 21.0
    assert total.shape == ()

    col_sum = m.sum(axis=0)
    expected_col = np.array([5, 7, 9], dtype=np.float32)
    assert np.array_equal(col_sum.data, expected_col)
    assert col_sum.shape == (3,)

    row_sum = m.sum(axis=1)
    expected_row = np.array([6, 15])
    assert np.array_equal(row_sum.data, expected_row)
    assert row_sum.shape == (2,)

    avg = m.mean()
    assert np.isclose(avg.data, 3.5)
    assert avg.shape == ()

    col_mean = m.mean(axis=0)
    expected_mean = np.array([2.5, 3.5, 4.5], dtype=np.float32)
    assert np.array_equal(col_mean.data, expected_mean)
    assert np.allclose(col_mean.data, expected_mean)

    maximum = m.max()
    assert maximum.data == 6
    assert maximum.shape == ()

    row_max = m.max(axis=1)
    expected_max = np.array([3, 6], dtype=np.float32)
    assert np.array_equal(row_max.data, expected_max)

    three_d = Tensor([[[1,2], [3,4]], [[5,6], [7,8]]])
    spatial_mean = three_d.mean(axis=(1, 2))
    assert spatial_mean.shape == (2,)

    print("✅ Reduction operations work correctly!")

if __name__ == "__main__":
    test_unit_reduction_operations()