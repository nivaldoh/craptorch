import numpy as np

from craptorch.core.tensor import Tensor
from craptorch.core.optimizers import Optimizer, Adam, AdamW

def test_unit_adamw_optimizer():
    param_adam = Tensor([1.0, 2.0], requires_grad=True)
    param_adamw = Tensor([1.0, 2.0], requires_grad=True)

    param_adam.grad = Tensor([0.1, 0.2])
    param_adamw.grad = Tensor([0.1, 0.2])

    adam = Adam([param_adam], lr=0.01, weight_decay=0.01)
    adamw = AdamW([param_adamw], lr=0.01, weight_decay=0.01)

    adam.step()
    adamw.step()

    assert not np.allclose(param_adam.data, param_adamw.data, rtol=1e-6)

    param = Tensor([1.0, 2.0], requires_grad=True)
    param.grad = Tensor([0.1, 0.2])

    optimizer = AdamW([param], lr=0.01, weight_decay=0.01)
    original_data = param.data.copy()

    optimizer.step()

    assert not np.array_equal(param.data, original_data)
    assert optimizer.step_count == 1

    assert optimizer.m_buffers[0] is not None
    assert optimizer.v_buffers[0] is not None

    # Test zero weight decay behaves like Adam
    param1 = Tensor([1.0, 2.0], requires_grad=True)
    param2 = Tensor([1.0, 2.0], requires_grad=True)

    param1.grad = Tensor([0.1, 0.2])
    param2.grad = Tensor([0.1, 0.2])

    adam_no_wd = Adam([param1], lr=0.01, weight_decay=0.0)
    adamw_no_wd = AdamW([param2], lr=0.01, weight_decay=0.0)

    adam_no_wd.step()
    adamw_no_wd.step()

    # Should be very similar (within numerical precision)
    assert np.allclose(param1.data, param2.data, rtol=1e-10)

if __name__ == "__main__":
    test_unit_adamw_optimizer()