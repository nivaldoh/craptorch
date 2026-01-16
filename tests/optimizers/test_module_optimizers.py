import numpy as np

from craptorch.core.tensor import Tensor
from craptorch.core.layers import Linear
from craptorch.core.activations import ReLU
from craptorch.core.losses import MSELoss
from craptorch.core.optimizers import SGD, Adam, AdamW

def test_module_optimizers():
    W1 = Tensor(np.random.randn(3,4) * 0.1, requires_grad=True)
    b1 = Tensor(np.zeros(4), requires_grad=True)

    W2 = Tensor(np.random.randn(4,2) * 0.1, requires_grad=True)
    b2 = Tensor(np.zeros(2), requires_grad=True)

    params = [W1, b1, W2, b2]

    W1.grad = Tensor(np.random.randn(3, 4) * 0.01)
    b1.grad = Tensor(np.random.randn(4) * 0.01)
    W2.grad = Tensor(np.random.randn(4, 2) * 0.01)
    b2.grad = Tensor(np.random.randn(2) * 0.01)

    # Test all optimizers on same network
    optimizers = [
        SGD(params, lr=0.01, momentum=0.9),
        Adam([p for p in params], lr=0.001),  # Fresh param list for Adam
        AdamW([p for p in params], lr=0.001, weight_decay=0.01)  # Fresh param list for AdamW
    ]

    original_params = [p.data.copy() for p in params]

    # Test SGD
    optimizers[0].step()
    sgd_params = [p.data.copy() for p in params]

    # Restore parameters and test Adam
    for i, p in enumerate(params):
        p.data = original_params[i].copy()
        # Re-add gradients since they may have been modified
        if i == 0:
            p.grad = Tensor(np.random.randn(3, 4) * 0.01)
        elif i == 1:
            p.grad = Tensor(np.random.randn(4) * 0.01)
        elif i == 2:
            p.grad = Tensor(np.random.randn(4, 2) * 0.01)
        else:
            p.grad = Tensor(np.random.randn(2) * 0.01)

    # Update parameter references for Adam
    optimizers[1].params = params
    optimizers[1].step()
    adam_params = [p.data.copy() for p in params]

    # Restore parameters and test AdamW
    for i, p in enumerate(params):
        p.data = original_params[i].copy()
        # Re-add gradients
        if i == 0:
            p.grad = Tensor(np.random.randn(3, 4) * 0.01)
        elif i == 1:
            p.grad = Tensor(np.random.randn(4) * 0.01)
        elif i == 2:
            p.grad = Tensor(np.random.randn(4, 2) * 0.01)
        else:
            p.grad = Tensor(np.random.randn(2) * 0.01)

    # Update parameter references for AdamW
    optimizers[2].params = params
    optimizers[2].step()
    adamw_params = [p.data.copy() for p in params]

    # Verify parameters changed differently for each optimizer
    for i in range(len(params)):
        # Parameters should be different from original
        assert not np.array_equal(sgd_params[i], original_params[i])
        assert not np.array_equal(adam_params[i], original_params[i])
        assert not np.array_equal(adamw_params[i], original_params[i])

        # Different optimizers should produce different results
        assert not np.allclose(sgd_params[i], adam_params[i], rtol=1e-6)

    # Test optimizer state management

    param = Tensor([1.0, 2.0], requires_grad=True)
    param.grad = Tensor([0.1, 0.2])

    optimizer = Adam([param], lr=0.001)

    # First step should initialize buffers
    optimizer.step()
    assert optimizer.m_buffers[0] is not None
    assert optimizer.v_buffers[0] is not None
    assert optimizer.step_count == 1

    # Zero grad should clear gradients but preserve optimizer state
    optimizer.zero_grad()
    assert param.grad is None
    assert optimizer.m_buffers[0] is not None  # State preserved
    assert optimizer.step_count == 1  # Step count preserved

if __name__ == "__main__":
    test_module_optimizers()