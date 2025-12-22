from craptorch.core.tensor import Tensor
from craptorch.core.activations import TOLERANCE, Tanh

import numpy as np

def test_unit_tanh():
    tanh = Tanh()

    x = Tensor([0.0])
    act = tanh.forward(x)
    assert np.allclose(act.data, [0.0])

    rang = Tensor([-10, -1, 0, 1, 10])
    act = tanh.forward(rang)
    assert np.all(act.data >=-1) and np.all(act.data <=1)

    sym = Tensor([2.0])
    act = tanh.forward(sym)
    sym_neg = Tensor([-2.0])
    act_neg = tanh.forward(sym_neg)
    assert np.allclose(act.data, -act_neg.data)

    extreme = Tensor([-1000, 1000])
    act = tanh.forward(extreme)
    assert np.allclose(act.data[0], -1, atol=TOLERANCE)
    assert np.allclose(act.data[1], 1, atol=TOLERANCE)

if __name__ == "__main__":
    test_unit_tanh()