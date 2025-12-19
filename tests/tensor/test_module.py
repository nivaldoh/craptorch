from craptorch.core.tensor import Tensor

import numpy as np

def test_module():
    print("🧪 RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    x = Tensor([[1, 2, 3], [4, 5, 6]])

    W1 = Tensor([[0.1, 0.2, 0.3, 0.4],
                 [0.5, 0.6, 0.7, 0.8],
                 [0.9, 1.0, 1.1, 1.2]])
    b1 = Tensor([0.1, 0.2, 0.3, 0.4])

    hidden = x.matmul(W1) + b1
    assert hidden.shape == (2,4), f"Expected (2,4), got {hidden.shape}"

    W2 = Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    b2 = Tensor([0.1, 0.2])

    output = hidden.matmul(W2) + b2
    assert output.shape == (2, 2), f"Expected (2, 2), got {output.shape}"

    assert not np.isnan(output.data).any(), "Output contains NaN values"
    assert np.isfinite(output.data).all(), "Output contains infinite values"

    print("✅ Two-layer neural network computation works!")

    print("🧪 Integration Test: Gradient System Readiness...")
    grad_tensor = Tensor([1, 2, 3], requires_grad=True)
    result = grad_tensor + 5
    assert grad_tensor.requires_grad == True, "requires_grad not preserved"
    assert grad_tensor.grad is None, "grad should still be None"

    grad_tensor.backward()

    print("🧪 Integration Test: Complex Shape Operations...")
    data = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    tensor_3d = data.reshape(2, 2, 3)
    assert tensor_3d.shape == (2,2,3)

    pooled = tensor_3d.mean(axis=(1,2))
    assert pooled.shape == (2,), f"Expected (2,), got {pooled.shape}"

    flattened = tensor_3d.reshape(2,-1)
    assert flattened.shape == (2, 6)

    transposed = tensor_3d.transpose()
    assert transposed.shape == (2, 3, 2)

    print("✅ Complex shape operations work!")

    print("🧪 Integration Test: Broadcasting Edge Cases...")
    scalar = Tensor(5.0)
    vector = Tensor([1, 2, 3])
    result = scalar + vector
    expected = np.array([6,7,8], dtype=np.float32)
    assert np.array_equal(result.data, expected)

    matrix = Tensor([[1, 2], [3, 4]])
    vec = Tensor([10, 20])
    result = matrix + vec
    expected = np.array([[11, 22], [13, 24]], dtype=np.float32)
    assert np.array_equal(result.data, expected)

    print("✅ Broadcasting edge cases work!")

if __name__ == "__main__":
    test_module()