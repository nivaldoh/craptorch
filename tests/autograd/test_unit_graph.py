import numpy as np

import craptorch.core.autograd  # ensure autograd patching is enabled
from craptorch.core.graph import to_dot
from craptorch.core.tensor import Tensor


def test_unit_graph_dot_contains_ops():
    x = Tensor([2.0], requires_grad=True)
    y = x * 3
    z = y + 1

    dot = to_dot(z)
    assert "MulBackward" in dot
    assert "AddBackward" in dot


def test_unit_backward_trace_records_grads():
    x = Tensor([2.0], requires_grad=True)
    y = x * 3
    z = y + 1

    trace = z.backward(trace=True)
    assert trace is not None
    assert trace.get("order")
    assert id(x) in trace.get("grads", {})
    assert np.allclose(trace["grads"][id(x)], [3.0])
