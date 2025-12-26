import numpy as np

from craptorch.core.tensor import Tensor
from craptorch.core.activations import ReLU, Sigmoid

XAVIER_SCALE_FACTOR = 1.0
HE_SCALE_FACTOR = 2.0

DROPOUT_MIN_PROB = 0.0
DROPOUT_MAX_PROB = 1.0

class Layer:
    def forward(self, x):
        raise NotImplementedError("subclasses must implement forward pass")

    def __call__(self, x, *args, **kwargs):
        return forward(x, *args, **kwargs)

    def parameters(self):
        return []

    def __repr(self):
        return f"{self.__class__.__name__}()"

class Linear(Layer):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features

        scale = np.sqrt(XAVIER_SCALE_FACTOR / in_features)
        weight_data = np.random.randn(in_features, out_features) * scale
        self.weight = Tensor(weight_data, requires_grad = True)

        if bias:
            bias_data = np.zeros(out_features)
            self.bias = Tensor(bias_data, requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
        y = x.matmul(self.weight)
        if self.bias is not None:
            y += self.bias
        return y

    def parameters(self):
        p = [self.weight]
        if self.bias is not None:
            p.append(self.bias)
        return p

    def __repr__(self):
        bias_str = f", bias={self.bias is not None}"
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}{bias_str})"