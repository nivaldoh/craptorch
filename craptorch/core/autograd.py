import numpy as np
from typing import Optional, List, Tuple
import sys
import os

from craptorch.core.tensor import Tensor

class Function:
    def __init__(self, *tensors):
        self.saved_tensors = list(tensors)
        self.next_functions = []

        # build computation graph connections
        for t in tensors:
            # check if tensor was created by another operation
            if isinstance(t, Tensor) and t.requires_grad:
                if getattr(t, '_grad_fn', None) is not None:
                    self.next_functions.append(t._grad_fn)
        
    def apply(self, grad_output):
        raise NotImplementedError("Implemented individually by each function")

##############################
# Ops
##############################

class AddBackward(Function):
    def apply(self, grad_output):
        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = grad_output

        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = grad_output

        return grad_a, grad_b

class MulBackward(Function):
    def apply(self, grad_output):
        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:
            if isinstance(b, Tensor):
                grad_a = b.data * grad_output
            else:
                grad_a = b * grad_output

        if isinstance(b, Tensor) and b.requires_grad:
            if isinstance(a, Tensor):
                grad_b = a.data * grad_output
            else:
                grad_b = a * grad_output

        return grad_a,grad_b

class SubBackward(Function):
    def apply(self, grad_output):
        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = grad_output

        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = -grad_output

        return grad_a, grad_b

class DivBackward(Function):
    def apply(self, grad_output):
        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:
            if isinstance(b, Tensor):
                grad_a = grad_output/b.data
            else:
                grad_a = grad_output/b

        if isinstance(b, Tensor) and b.requires_grad:
            # we just assume "a" is a tensor here?
            a_val = a.data if isinstance(a, Tensor) else a
            grad_b = -grad_output * a_val/(b.data ** 2)

        return grad_a, grad_b

class MatMulBackward(Function):
    def apply(self, grad_output):
        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:
            # For batched tensors, transpose only last two dims (why?)
            if b.data.ndim >= 2:
                b_T = np.swapaxes(b.data, -2, -1)
            else:
                b_T = b.data.T
            grad_a = np.matmul(grad_output, b_T)

        if isinstance(b, Tensor) and b.requires_grad:
            if b.data.ndim >= 2:
                a_T = np.swapaxes(a.data, -2, -1)
            else:
                a_T = a.data.T
            grad_b = np.matmul(a_T, grad_output)

        return grad_a, grad_b

class TransposeBackward(Function):
    def __init__(self, tensor, dim0, dim1):
        super().__init__(tensor)
        self.dim0 = dim0
        self.dim1 = dim1

    def apply(self, grad_output):
        x, = self.saved_tensors
        grad_x = None

        if isinstance(x, Tensor) and x.requires_grad:
            if self.dim0 is None and self.dim1 is None:
                if grad_output.ndim < 2:
                    grad_x = grad_output.copy()
                else:
                    axes = list(range(grad_output.ndim))
                    axes[-2], axes[-1] = axes[-1], axes[-2]
                    grad_x = np.transpose(grad_output, axes)
            else:
                axes = list(range(grad_output.ndim))
                axes[self.dim0], axes[self.dim1] = axes[self.dim1], axes[self.dim0]
                grad_x = np.transpose(grad_output, axes)
        
        return (grad_x,)

class PermuteBackward(Function):
    def __init__(self, tensor, axes):
        super().__init__(tensor)
        self.axes = axes
        self.inverse_axes = tuple(np.argsort(axes))

    def apply(self, grad_output):
        x, = self.saved_tensors
        grad_x = None

        if isinstance(x, Tensor) and x.requires_grad:
            grad_x = np.transpose(grad_output, self.inverse_axes)

        return (grad_x,)

class EmbeddingBackward(Function):
    def __init__(self, weight, indices):
        super().__init__(weight)
        self.indices = indices

    def apply(self, grad_output):
        weight, = self.saved_tensors
        grad_weight = None

        if isinstance(weight, Tensor) and weight.requires_grad:
            grad_weight = np.zeros_like(weight.data)

            # scatter grads back to weights
            # np.add.at accumulates grads for repeated indices
            indices_flat = self.indices.data.astype(int).flatten()
            grad_output_reshaped = grad_output.reshape(-1, grad_output.shape[-1])

            np.add.at(grad_weight, indices_flat, grad_output_reshaped)

        return (grad_weight,)

class SliceBackward(Function):
    def __init__(self, tensor, key):
        super().__init__(tensor)
        self.key = key
        self.original_shape = tensor.shape

    def apply(self, grad_output):
        tensor, = self.saved_tensors
        grad_input = None

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            grad_input = np.zeros(self.original_shape, dtype=np.float32)

            # propagate grads only to the positions that impacted the loss
            grad_input[self.key] = grad_output
        
        return (grad_input,)

class ReshapeBackward(Function):
    def __init__(self, tensor, original_shape):
        super().__init__(tensor)
        self.original_shape = original_shape

    def apply(self, grad_output):
        x, = self.saved_tensors
        grad_x = None

        if isinstance(x, Tensor) and x.requires_grad:
            grad_x = grad_output.reshape(self.original_shape)
        
        return (grad_x,)

class SumBackward(Function):
    def apply(self, grad_output):
        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            return (np.ones_like(tensor.data) * grad_output,)

        return None,

##############################
# Activations
##############################

class ReLUBackward(Function):
    def __init__(self, input_tensor):
        super().__init__(input_tensor)

    def apply(self, grad_output):
        x = self.saved_tensors

        if isinstance(x, Tensor) and x.requires_grad:
            relu_grad = (x.data > 0).astype(np.float32)
            return grad_output * relu_grad,

        return (None,)

class SigmoidBackward(Function):
    def __init__(self, input_tensor, output_tensor):
        # store original input to sigmoid
        super().__init__(input_tensor)
        # store original output from sigmoid
        self.output_data = output_tensor.data

    def apply(self, grad_output):
        x, = self.saved_tensors
        grad_x = None

        if isinstance(x, Tensor) and x.requires_grad:
            sigmoid_grad = self.output_data * (1 - self.output_data)
            return grad_output * sigmoid_grad

        return (None,)

class SofmaxBackward(Function):
    def __init__(self, input_tensor, output_tensor, dim=-1):
        super().__init__(input_tensor)
        self.output_data = output_tensor.data
        self.dim=dim

    def apply(self, grad_output):
        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            sum_term = np.sum(grad_output * self.output_data, axis=self.dim, keepdims=True)

            grad_x = self.output_data * (grad_output - sum_term)
            return (grad_x,)
        return (None,)

class GeLUBackward(Function):
    def __init__(self, input_tensor):
        super().__init__(input_tensor)

    def apply(self, grad_output):
        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            x = tensor.data

            # derivative approximation
            sqrt_2_over_pi = np.sqrt(2.0/np.pi)
            x_cubed = x ** 3
            tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
            tanh_out = np.tanh(tanh_arg)
            sech_squared = 1 - tanh_out ** 2

            d_tanh_arg = sqrt_2_over_pi * (1 + 0.134145 * x ** 2)
            gelu_grad = 0.5 * (1 + tanh_out) + 0.5 * x * sech_squared * d_tanh_arg
            
            return (grad_output * gelu_grad)

        return (None,)

##############################
# Losses
##############################

class MSEBackward(Function):
    def __init__(self, predictions, targets):
        super().__init__(predictions)
        self.targets_data = targets.data
        self.num_samples = np.size(targets.data)

    def apply(self, grad_output):
        predictions, = self.saved_tensors

        if isinstance(predictions, Tensor) and predictions.requires_grad:
            grad = 2.0 * (predictions.data - self.targets_data) / self.num_samples

            return grad * grad_output,
        return None,

class BCEBackward(Function):
    def __init__(self, predictions, targets):
        super().__init__(predictions)
        self.targets_data = targets.data
        self.num_samples = np.size(targets.data)

    def apply(self, grad_output):
        predictions, = self.saved_tensors

        if isinstance(predictions, Tensor) and predictions.requires_grad:
            p = np.clip(predictions.data, EPSILON, 1-EPSILON)
            y = self.targets.data

            grad = (p-y)/(p*(1-p)*self.num_samples)

            return grad * grad_output

        return None,

class CrossEntropyBackward(Function):
    def __init__(self, logits, targets):
        super().__init__(logits)
        self.targets_data = targets.data.astype(int)
        self.batch_size = logits.data.shape[0]
        self.num_classes = logits.data.shape[1]

    def apply(self, grad_output):
        logits, = self.saved_tensors

        if isinstance(logits, Tensor) and logits.requires_grad:
            logits_data = logits.data
            max_logits = np.max(logits_data, axis=1, keepdims=True)
            exp_logits = np.exp(logits_data - max_logits)
            softmax = exp_logits/np.sum(exp_logits, axis=1, keepdims=True)

            one_hot = np.zeros((self.batch_size, self.num_classes), dtype=np.float32)
            one_hot[np.arange(self.batch_size), self.targets_data] = 1.0

            grad = (softmax - one_hot)/self.batch_size

            return grad * grad_output,
        return None,

def enable_autograd(quiet=False):
    1