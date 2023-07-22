from __future__ import annotations
from functools import wraps

import numpy as np


# These decorators reduce Tensor methods into returning (1) the output data and (2)
# "local" gradient(s)
def _single_var(method):
    @wraps(method)
    def wrapper(self: Tensor, *args, **kwargs):
        data, grad = method(self, *args, **kwargs)
        out = Tensor(data)  # we need to reference this object for the chain_rule
        out._inputs = {self}

        def chain_rule():  # assume out.grad is set correctly
            self.grad += out.grad * grad

        out._backward = chain_rule
        return out

    return wrapper


def _double_var(is_elt_wise: bool = True):
    # is_elt_wise = False is for matrix dot products
    def decorator(method):
        @wraps(method)
        def wrapper(self: Tensor, other: Tensor, *args, **kwargs):
            data, self_grad, other_grad = method(self, other, *args, **kwargs)
            out = Tensor(data)  # we need to reference this object for the chain_rule
            out._inputs = {self, other}

            # self and other could be vectors (shape is (d,) not (d, 1)) or scalars.
            # in these cases, the correct operator to combine gradients via the chain
            # rule is multiplication, not the dot product
            nonlocal is_elt_wise  # lemme modify it based on data.shape
            if not is_elt_wise:
                if len(data.shape) == 0:  # scalar = vector-vector dot product
                    is_elt_wise = True
                elif len(data.shape) == 1:  # vector = matrix-vector dot product
                    is_elt_wise = data.shape[0] == 1

            def chain_rule():  # assume out.grad is set correctly
                if is_elt_wise:
                    self.grad += out.grad * self_grad
                    other.grad += out.grad * other_grad
                else:  # tricky
                    self.grad += out.grad @ self_grad
                    other.grad += other_grad @ out.grad

            out._backward = chain_rule
            return out

        return wrapper

    return decorator


class Tensor:
    def __init__(self, data: list[float]):
        self._data: np.ndarray = np.array(data)
        self._inputs: set[Tensor] = set()  # tensors which this tensor is a function of
        self._backward = lambda: None  # backward pass function
        self.grad = (
            0  # only the root's grad is 1, rest 0 until the root is a function of it
        )

    @_double_var()
    def __add__(self, other: Tensor) -> Tensor:
        data = self._data + other._data
        self_grad = 1
        other_grad = 1
        return data, self_grad, other_grad

    @_double_var()
    def __mul__(self, other: Tensor) -> Tensor:
        data = self._data * other._data
        self_grad = other._data
        other_grad = self._data
        return data, self_grad, other_grad

    @_double_var(is_elt_wise=False)
    def dot(self, other: Tensor) -> Tensor:
        data = self._data @ other._data
        # the following 2 lines of code took me a while to figure out. i dont think i
        # even learned anything interesting during that struggle lol
        self_grad = other._data.T
        other_grad = self._data.T
        return data, self_grad, other_grad

    @_single_var
    def __pow__(self, exponent: float | int) -> Tensor:
        if not isinstance(exponent, (float, int)):
            raise TypeError(f"exponent must be a float/constant. Got {exponent}.")
        data = self._data**exponent
        grad = exponent * self._data ** (exponent - 1)
        return data, grad

    @_single_var
    def sum(self, dim: int | None = None) -> Tensor:
        data = self._data.sum(axis=dim)
        grad = np.ones_like(self._data)
        return data, grad

    @_single_var
    def __getitem__(self, key) -> Tensor:
        data = self._data[key]
        grad = np.zeros_like(self._data)
        grad[key] = 1
        return data, grad

    @property
    @_single_var
    def T(self) -> Tensor:
        data = self._data.T
        grad = np.ones_like(self.grad.T) if isinstance(self.grad, np.ndarray) else 1
        return data, grad

    def concat(self, other: Tensor, dim: int = -1) -> Tensor:
        raise NotImplementedError

    @_single_var
    def exp(self) -> Tensor:
        data = np.exp(self._data)
        grad = data
        return data, grad

    @_single_var
    def log(self) -> Tensor:
        data = np.log(self._data)
        grad = 1 / self._data
        return data, grad

    @_single_var
    def relu(self) -> Tensor:
        data = np.maximum(0, self._data)
        grad = (data > 0).astype(self._data.dtype)
        return data, grad

    def cross_entropy(self, labels: list[int]) -> Tensor:
        # self._data are logits. labels are sparsely encoded
        raise NotImplementedError

    @_single_var
    def sigmoid(self) -> Tensor:
        data = 1 / (1 + np.exp(-self._data))
        grad = data * (1 - data)
        return data, grad

    @_single_var
    def log_sigmoid(self) -> Tensor:
        data = -np.logaddexp(0, -self._data)
        grad = 1 - np.exp(data)
        return data, grad

    def log_softmax(self, dim: int = -1) -> Tensor:
        raise NotImplementedError

    def negative_log_likelihood(self, labels: list[int]) -> Tensor:
        raise NotImplementedError

    def __matmul__(self, other: Tensor) -> Tensor:
        return self.dot(other)

    def __rmul__(self, other: Tensor | float | int) -> Tensor:
        if isinstance(other, (float, int)):
            other = Tensor([other])
        return self * other  # just call the left-multiply, i.e., __mul__, method

    def __neg__(self) -> Tensor:
        return self * Tensor([-1])

    def __sub__(self, other: Tensor | float | int) -> Tensor:
        return self + (-other)

    def __truediv__(self, other: Tensor | float | int) -> Tensor:
        return self * Tensor([other**-1])

    def __rtruediv__(self, other: Tensor | float | int) -> Tensor:
        return other * self**-1

    @property
    def shape(self):
        return self._data.shape

    def item(self):
        if not isinstance(self._data, np.ndarray):
            return self._data
        else:
            raise ValueError("This Tensor has more than one item.")

    def __repr__(self) -> str:
        # return the numpy array's repr but replace "array" with "Tensor", adjusting
        # whitespace for the difference in the length of the names
        numpy_name = "array"
        class_name = self.__class__.__name__
        array_strings = (
            repr(self._data)
            .removeprefix(f"{numpy_name}(")
            .removesuffix(")")
            .split("\n")
        )
        whitespace = " " * (len(class_name) - len(numpy_name))  # assume positive
        data_repr = (
            array_strings[0]
            + "\n"
            + "\n".join(whitespace + array_string for array_string in array_strings[1:])
        )
        if len(self.shape) <= 1:  # it's a 0-D or 1-D array
            data_repr = data_repr.rstrip("\n")
        return f"{class_name}({data_repr})"

    def backward(self):
        # reverse-topological order, i.e., root -> leaves.
        # that's b/c, for the chain rule, we can only define the gradient of the root
        # wrt some node if we know the "local gradient" and the gradient of the root wrt
        # the node's parent (we multiply the two). so we need to calculate gradients
        # from parents -> children or root -> leaves.
        def _reverse_topo_order(tensor: Tensor) -> list[Tensor]:
            # strategy: regular topological order, and then reverse
            nodes_ordered: list[Tensor] = []
            _visited: set[Tensor] = set()

            def dfs(node: Tensor):
                _visited.add(node)
                # topological order invariant: a node's children must be appended before
                # appending itself
                for child in node._inputs:
                    if child not in _visited:
                        dfs(child)
                nodes_ordered.append(node)

            dfs(tensor)  # nodes_ordered is leaves -> root
            nodes_ordered.reverse()
            return nodes_ordered

        nodes_ordered = _reverse_topo_order(self)

        nodes_ordered[0].grad = 1  # derivative root wrt root. it was initialized to 0
        for node in nodes_ordered:
            node._backward()
