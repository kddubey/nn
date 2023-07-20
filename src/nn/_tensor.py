from __future__ import annotations

import numpy as np


def _reverse_topo_order(tensor: Tensor) -> list[Tensor]:
    # strategy: regular topological order, and then reverse
    nodes_ordered: list[Tensor] = []
    _visited: set[Tensor] = set()

    def dfs(node: Tensor):
        _visited.add(node)
        # regular topological order invariant: a node's children must be appended before
        # appending itself
        for child in node._inputs:
            if child not in _visited:
                dfs(child)
        nodes_ordered.append(node)

    dfs(tensor)  # nodes_ordered is leaves -> root
    nodes_ordered.reverse()

    return nodes_ordered


class Tensor:
    def __init__(self, data: list[float]):
        self._data: np.ndarray = np.array(data)
        self._inputs: set[Tensor] = set()  # tensors which this tensor is a function of
        self._backward = lambda: None  # backward pass function
        self.grad = (
            0  # only the root's grad is 1, rest 0 until the root is a function of it
        )

    @property
    def shape(self):
        return self._data.shape

    @property
    def T(self):
        if self._inputs:
            raise ValueError("Narrrgo!")
        self._data = self._data.T
        return self

    def backward(self):
        # reverse-topological order, i.e., root -> leaves.
        # that's b/c, for the chain rule, we can only define the gradient of the root
        # wrt some node if we know the "local gradient" and the gradient of the root wrt
        # the node's parent (we multiply the two). so we need to calculate gradients
        # from parents -> children or root -> leaves.
        nodes_ordered = _reverse_topo_order(self)

        nodes_ordered[0].grad = 1  # derivative root wrt root. it was initialized to 0
        for node in nodes_ordered:
            node._backward()

    def __add__(self, other: Tensor) -> Tensor:
        # forward pass
        data = self._data + other._data

        # output Tensor which we'll need to reference for backward
        out = Tensor(data)
        out._inputs = {self, other}

        # define local gradients
        self_grad = 1
        other_grad = 1

        # define backward pass for global gradient calculation, assuming we know the
        # output's gradient
        def backward():
            self.grad += out.grad * self_grad
            other.grad += out.grad * other_grad

        out._backward = backward
        return out

    def __mul__(self, other: Tensor) -> Tensor:
        data = self._data * other._data
        out = Tensor(data)
        out._inputs = {self, other}

        self_grad = other._data
        other_grad = self._data

        def backward():
            self.grad += out.grad * self_grad
            other.grad += out.grad * other_grad

        out._backward = backward
        return out

    def dot(self, other: Tensor) -> Tensor:
        data = self._data @ other._data
        out = Tensor(data)
        out._inputs = {self, other}

        self_grad = other._data.T
        other_grad = self._data.T

        def backward():
            self.grad += out.grad @ self_grad
            other.grad += other_grad @ out.grad

        out._backward = backward
        return out

    def relu(self) -> Tensor:
        data = np.maximum(0, self._data)
        out = Tensor(data)
        out._inputs = {self}

        self_grad = (data > 0).astype(self._data.dtype)

        def backward():
            self.grad += out.grad * self_grad

        out._backward = backward
        return out

    def sigmoid(self) -> Tensor:
        data = 1 / (1 + np.exp(-self._data))
        out = Tensor(data)
        out._inputs = {self}

        self_grad = data * (1 - data)

        def backward():
            self.grad += out.grad * self_grad

        out._backward = backward
        return out

    def log_softmax(self, dim: int = -1) -> Tensor:
        raise NotImplementedError

    def negative_log_likelihood(self, label: int) -> Tensor:
        # input checks
        if label not in {0, 1}:
            raise ValueError("label must be 0 or 1.")
        if self._data <= 0 or self._data >= 1:
            raise ValueError("data must be a probability in (0, 1).")

        if label == 0:
            data = -np.log(1 - self._data)
        else:  # it's 1
            data = -np.log(self._data)

        out = Tensor(data)
        out._inputs = {self}

        if label == 0:
            self_grad = 1 / self._data
        else:  # it's 1
            self_grad = 1 / (1 - self._data)

        def backward():
            self.grad += out.grad * self_grad

        out._backward = backward
        return out

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
        if len(self.shape) == 1:  # it's a 1-D array
            data_repr = data_repr.rstrip("\n")
        return f"{class_name}({data_repr})"
