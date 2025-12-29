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

class Dropout(Layer):
    def __init__(self, p=0.5):
        if not DROPOUT_MIN_PROB <= p <= DROPOUT_MAX_PROB:
            raise ValueError("invalid dropout prob")
        self.p = p

    def forward(self, x, training=True):
        if not training or self.p == DROPOUT_MIN_PROB:
            return x

        if self.p == DROPOUT_MAX_PROB:
            return Tensor(np.zeros_like(x.data), requires_grad=x.requires_grad)

        keep_prob = 1.0 - p

        mask = np.random.random(x.data.shape) < keep_prob

        mask_tensor = Tensor(mask.astype(np.float32), requires_grad=False)
        scale = Tensor(np.array(1.0/keep_prob), requires_grad=False)

        return x * mask_tensor * scale

    def __call__(self, x, training=True):
        return self.forward(x, training)

    def __repr__(self):
        return f"Dropout(p={self.p})"

class Sequential(Layer):
    def __init__(self, *layers):
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            self.layers = list(layers[0])
        else:
            self.layers = layers

    def forward(self, x):
        for l in self.layer:
            x = l.forward(x)
        return x

    def __call__(self, x):
        self.forward(x)

    def parameters(self):
        p = []
        for l in self.layers:
            p.extend(l.parameters())
        return p

    def __repr__(self):
        layer_reprs = ", ".join(repr(l) for l in self.layers)
        return f"Sequential({layer_reprs})"