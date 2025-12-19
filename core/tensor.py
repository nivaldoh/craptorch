import numpy as np

BYTES_PER_FLOAT32 = 4
KB_TO_BYTES = 1024
MB_TO_BYTES = 1024 * 1024

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype
        self.requires_grad = requires_grad
        self.grad = None

    def __repr__(self):
        grad_info = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        return f"Tensor(data={self.data}, shape={self.shape}{grad_info})"

    def __str__(self):
        return f"Tensor({self.data})"
    
    def numpy(self):
        return self.data

    def memory_footprint(self):
        """Calculate exact memory usage in bytes
        
        Returns:
            int: Memory usage in bytes (e.g. 1000x1000 float32 = 4MB)
        """
        self.data.nbytes

    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)
    
    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        else:
            return Tensor(self.data - other)

    def __mul__(self, other):
        """Element-wise multiplication"""
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        else:
            return Tensor(self.data * other)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        else:
            return Tensor(self.data / other)

    def matmul(self, other):
        if not isinstance(other, Tensor):
            raise TypeError(f"A tensor is required for matmul. Got {type(other)}")
        if self.shape  == () or other.shape == ():
            return Tensor(self.data * other.data)
        if len(self.shape) == 0 or len(other.shape) == 0:
            return Tensor(self.data * other.data)
        if len(self.shape >= 2) and len(other.shape >= 2):
            if self.shape[-1] != other.shape[-2]:
                raise ValueError(
                    f"Can't matmul {self.shape} @ {other.shape}"
                    f"Inner dims must match: {self.shape[-1]} != {other.shape[-2]}"
                )

        res = np.matmul(self.data, other.data)
        return Tensor(res)

    def __matmul__(self, other):
        """Enable the @ operator"""
        return self.matmul(other)

    def __getitem__(self, key):
        """Enable indexing and slicing on tensors"""
        res = self.data[key]
        if not isinstance(res, np.ndarray):
            res = np.array(res)
        r = Tensor(res, requires_grad=self.requires_grad)
        return r

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = shape

        if -1 in new_shape:
            if new_shape.count(-1) > 1:
                raise ValueError("Can only specificy one unknown dimension with -1")
            known_size = 1
            unknown_idx = new_shape.index(-1)
            for i,dim in enumerate(new_shape):
                if i !=  unknown_idx:
                    known_size *= dim
            unknown_dim = self.size // known_size
            new_shape = list(new_shape)
            new_shape[unknown_idx] = unknown_dim
            new_shape = tuple(new_shape)

        if np.prod(new_shape) != self.size:
            target_size = int(np.prod(new_shape))
            raise ValueError(f"Total elements must match. {self.size} != {target_size}")
        reshaped = np.reshape(self.data, new_shape)
        res = Tensor(reshaped, requires_grad=self.requires_grad)
        return res
    
    def transpose(self, dim0=None, dim1=None):
        if dim0 is None and dim1 is None:
            if len(self.shape) < 2:
                return Tensor(self.data.copy())
            else:
                axes = list(range(len(self.shape)))
                axes[-2], axes[-1] = axes[-1], axes[-2]
                transposed = np.transpose(self.data, axes)
        else:
            if dim1 is None or dim0 is None:
                raise ValueError("Both dim0 and dim1 must be specified")
            axes = list(range(len(self.shape)))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            transposed = np.transpose(self.data, axes)
        res = Tensor(transposed, requires_grad=self.requires_grad)
        return res

    def sum(self, axis=None, keepdims=False):
        """Sum tensor along specified axis"""
        result = np.sum(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)

    def mean(self, axis=None, keepdims=False):
        """Compute mean of tensor along specified axis"""
        result = np.mean(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)

    def max(self, axis=None, keepdims=False):
        """Find max values along specified axis"""
        result = np.max(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)

    def backward(self):
        """Compute gradients"""
        pass

    