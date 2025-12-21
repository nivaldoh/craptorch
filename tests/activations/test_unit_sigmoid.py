from craptorch.core.tensor import Tensor
from craptorch.core.activations import TOLERANCE, Sigmoid

import numpy as np

def test_unit_sigmoid():
    print("🔬 Unit Test: Sigmoid...")

    sigmoid = Sigmoid()

    x = Tensor([0.0])
    result = sigmoid.forward(x)
    assert np.allclose(result.data, [0.5]), f"sigmoid should be 0.5 but got {result.data}"

    x = Tensor([-10, -1, 0, 1, 10])
    res = sigmoid.forward(x)
    assert np.all(res.data > 0) and np.all(res.data < 1)

    x = Tensor([-1000, 1000])  # Extreme values
    result = sigmoid.forward(x)

    assert np.allclose(result.data[0], 0, atol=TOLERANCE), f"sigmoid(-inf) should approach 0, got {result.data[0]}"
    assert np.allclose(result.data[1], 1, atol=TOLERANCE), f"sigmoid(+inf) should approach 1, got {result.data[1]}"

    print("✅ Sigmoid works correctly!")

if __name__ == "__main__":
    test_unit_sigmoid()