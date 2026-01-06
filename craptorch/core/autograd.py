import numpy as np
from typing import Optional, List, Tuple
import sys
import os

from craptorch.core.tensor import Tensor
from craptorch.core.losses import EPSILON

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
        x, = self.saved_tensors

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
            return grad_output * sigmoid_grad,

        return (None,)

class SoftmaxBackward(Function):
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
            
            return (grad_output * gelu_grad,)

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
            y = self.targets_data

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
    if hasattr(Tensor, '_autograd_enabled'):
        return

    _original_add = Tensor.__add__
    _original_sub = Tensor.__sub__
    _original_mul = Tensor.__mul__
    _original_div = Tensor.__truediv__
    _original_getitem = Tensor.__getitem__

    _original_matmul = Tensor.matmul
    _original_transpose = Tensor.transpose
    _original_reshape = Tensor.reshape

    def tracked_add(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        result = _original_add(self, other)

        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            result.requires_grad = True
            result._grad_fn = AddBackward(self, other)

        return result
    
    def tracked_mul(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        result = _original_mul(self, other)

        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            result.requires_grad = True
            result._grad_fn = MulBackward(self, other)

        return result

    def tracked_matmul(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        result = _original_matmul(self, other)

        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            result.requires_grad = True
            result._grad_fn = MatMulBackward(self, other)

        return result

    def tracked_transpose(self, dim0=None, dim1=None):
        result = _original_transpose(self, dim0, dim1)

        if self.requires_grad:
            result.requires_grad = True
            result._grad_fn = TransposeBackward(self, dim0, dim1)

        return result

    def tracked_reshape(self, *shape):
        original_shape = self.shape

        result = _original_reshape(self, *shape)

        if self.requires_grad:
            result.requires_grad = True
            result._grad_fn = ReshapeBackward(self, original_shape)

        return result

    def tracked_sub(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        result = _original_sub(self, other)

        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            result.requires_grad = True
            result._grad_fn = SubBackward(self, other)

        return result

    def tracked_div(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        result = _original_div(self, other)

        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            result.requires_grad = True
            result._grad_fn = DivBackward(self, other)

        return result

    def tracked_getitem(self, key):
        result = _original_getitem(self, key)

        if self.requires_grad:
            result.requires_grad = True
            result._grad_fn = SliceBackward(self, key)

        return result

    def sum_op(self, axis=None, keepdims=False):
        result_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        result = Tensor(result_data)

        if self.requires_grad:
            result.requires_grad = True
            result._grad_fn = SumBackward(self)

        return result

    def backward(self, gradient=None, trace=False, _trace=None):
        top_level = False
        if trace and _trace is None:
            _trace = {"order": [], "grads": {}}
            top_level = True

        if not self.requires_grad:
            return

        # initialize grad if not provided (for scalar outputs)
        if gradient is None:
            if self.data.size == 1:
                gradient = np.ones_like(self.data)
            else:
                raise ValueError("backward() called on non-scalar tensor without gradient arg.")

        # init or accumulate grad
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

        # handle broadcasting: sum gradient to match self.data shape
        if gradient.shape != self.grad.shape:
            # remove extra leading dims like batch size
            while gradient.ndim > self.grad.ndim:
                gradient = gradient.sum(axis=0)
            
            # sum over dimensions that were size-1 in the original tensor
            # e.g. broadcast bias to the entire batch
            for i in range(gradient.ndim):
                if self.grad.shape[i] == 1 and gradient.shape[i] != 1:
                    gradient = gradient.sum(axis=i, keepdims=True)

        if trace:
            _trace["order"].append({"tensor": self, "grad": gradient.copy()})
            _trace["grads"][id(self)] = gradient.copy()

        self.grad += gradient

        # propagate grads through computation graph
        # grad_fn is set by autograd enhancement when tensor is created from an operation
        grad_fn = getattr(self, "_grad_fn", None)
        if grad_fn is not None:
            grads = grad_fn.apply(gradient)

            # recursively call backward on parent tensors
            for tensor, grad in zip(grad_fn.saved_tensors, grads):
                if isinstance(tensor, Tensor) and tensor.requires_grad and grad is not None:
                    tensor.backward(grad, trace=trace, _trace=_trace)

        if top_level:
            return _trace

    def zero_grad(self):
        self.grad = None

    # install extended ops
    Tensor.__add__ = tracked_add
    Tensor.__sub__ = tracked_sub
    Tensor.__mul__ = tracked_mul
    Tensor.__truediv__ = tracked_div
    Tensor.__getitem__ = tracked_getitem
    Tensor.matmul = tracked_matmul
    Tensor.transpose = tracked_transpose
    Tensor.reshape = tracked_reshape
    Tensor.sum = sum_op
    Tensor.backward = backward
    Tensor.zero_grad = zero_grad

    from craptorch.core.activations import Sigmoid, ReLU, Softmax, GELU
    from craptorch.core.losses import BinaryCrossEntropyLoss, MSELoss, CrossEntropyLoss

    _original_sigmoid_forward = Sigmoid.forward
    _original_relu_forward = ReLU.forward
    _original_softmax_forward = Softmax.forward
    _original_gelu_forward = GELU.forward
    _original_bce_forward = BinaryCrossEntropyLoss.forward
    _original_mse_forward = MSELoss.forward
    _original_ce_forward = CrossEntropyLoss.forward

    def tracked_sigmoid_forward(self, x):
        result = _original_sigmoid_forward(self, x)

        if x.requires_grad:
            result.requires_grad = True
        result._grad_fn = SigmoidBackward(x, result)

        return result

    def tracked_relu_forward(self, x):
        result_data = np.maximum(0, x.data)
        result = Tensor(result_data)

        if x.requires_grad:
            result.requires_grad = True
            result._grad_fn = ReLUBackward(x)

        return result

    def tracked_softmax_forward(self, x, dim=-1):
        result = _original_softmax_forward(self, x, dim=dim)

        if x.requires_grad:
            result.requires_grad = True
            result._grad_fn = SoftmaxBackward(x, result, dim)

        return result

    def tracked_gelu_forward(self, x):
        result = _original_gelu_forward(self, x)

        if x.requires_grad:
            result.requires_grad = True
            result._grad_fn = GeLUBackward(x)
        
        return result

    def tracked_bce_forward(self, predictions, targets):
        clamped_preds = np.clip(predictions.data, EPSILON, 1-EPSILON)
        log_preds = np.log(clamped_preds)
        log_one_minus_preds = np.log(1 - clamped_preds)
        bce_per_sample = -(targets.data * log_preds + (1-targets.data)*log_one_minus_preds)
        bce_loss = np.mean(bce_per_sample)

        result = Tensor(bce_loss)

        if predictions.requires_grad:
            result.requires_grad = True
            result._grad_fn = BCEBackward(predictions, targets)

        return result

    def tracked_mse_forward(self, predictions, targets):
        diff = predictions.data - targets.data
        squared_diff = diff ** 2
        mse = np.mean(squared_diff)

        result = Tensor(mse)

        if predictions.requires_grad:
            result.requires_grad = True
            result._grad_fn = MSEBackward(predictions, targets)

        return result

    def tracked_ce_forward(self, logits, targets):
        from craptorch.core.losses import log_softmax

        log_probs = log_softmax(logits, dim=-1)

        batch_size = logits.shape[0]
        target_indices = targets.data.astype(int)
        selected_log_probs = log_probs.data[np.arange(batch_size), target_indices]

        ce_loss = -np.mean(selected_log_probs)

        result = Tensor(ce_loss)

        if logits.requires_grad:
            result.requires_grad = True
            result._grad_fn = CrossEntropyBackward(logits, targets)

        return result

    # Install extended activations and losses
    Sigmoid.forward = tracked_sigmoid_forward
    ReLU.forward = tracked_relu_forward
    Softmax.forward = tracked_softmax_forward
    GELU.forward = tracked_gelu_forward
    BinaryCrossEntropyLoss.forward = tracked_bce_forward
    MSELoss.forward = tracked_mse_forward
    CrossEntropyLoss.forward = tracked_ce_forward

    Tensor._autograd_enabled = True

    if not quiet:
        print('autograd enabled')

# Auto-enable when module is imported
# Always quiet to avoid cluttering user imports
import os
enable_autograd(quiet=True)
