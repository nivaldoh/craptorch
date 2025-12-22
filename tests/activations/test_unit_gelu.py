from craptorch.core.tensor import Tensor
from craptorch.core.activations import TOLERANCE, GELU

import numpy as np

def test_unit_gelu():
    gelu = GELU()

    zro = Tensor([0.0])
    act = gelu.forward(zro)
    assert np.allclose(act.data, [0.0], atol=TOLERANCE)

    pos = Tensor([1.0])
    act = gelu.forward(pos)
    assert act.data > 0.8

    neg = Tensor([-1.0])
    act = gelu.forward(neg)
    assert act.data < 0 and act.data[0] > -0.2

    x = Tensor([-0.001, 0.0, 0.001])
    act = gelu.forward(x)
    diff1 = abs(act.data[1] - act.data[0])
    diff2 = abs(act.data[2] - act.data[1])
    assert diff1 < 0.01 and diff2 < 0.01

if __name__ == "__main__":
    test_unit_gelu()