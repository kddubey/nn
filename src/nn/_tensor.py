from __future__ import annotations
from functools import partial, wraps

import numpy as np


def _log_sum_exp(
    array: np.ndarray,
    dim: int | None = None,
    keepdims: bool = False,
) -> np.ndarray:
    # adapted from scipy:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html
    array_maxs: np.ndarray = np.amax(array, axis=dim, keepdims=True)

    if array_maxs.ndim > 0:
        array_maxs[~np.isfinite(array_maxs)] = 0
    elif not np.isfinite(array_maxs):
        array_maxs = 0

    tmp = np.exp(array - array_maxs)

    # suppress warnings about log of zero
    with np.errstate(divide="ignore"):
        summed = np.sum(tmp, axis=dim, keepdims=keepdims)
        out = np.log(summed)

    if not keepdims:
        array_maxs = np.squeeze(array_maxs, axis=dim)
    out += array_maxs

    return out


def _force_transpose(array: np.ndarray) -> np.ndarray:
    # it's easier for me to code up the chain-rule-via-dot-product (Jacobian-gradient
    # dot product) when we explicitly transpose the "local" gradients, as we do in the
    # math
    if len(np.shape(array)) == 1:
        return array[:, np.newaxis].T
    return array.T


def _dot_and_maybe_squeeze(
    og_shape: tuple[int],
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    # if the "local" gradient was transposed from a numpy vector (n,) to a numpy matrix
    # (1, n), then we should ensure that the gradient has shape (n,). o.w. tensor -
    # tensor.grad broadcasts incorrectly. I basically treat this fxn as "do the dot
    # product that lines up w/ the math"
    z = x @ y
    if len(z.shape) > len(og_shape):
        z = z.squeeze(-1)
    return z


def _dot_product_chain_rule(
    self: Tensor,
    other: Tensor,
    out: Tensor,
    self_grad: np.ndarray,
    other_grad: np.ndarray,
) -> None:
    # I need to overcome some limitations of numpy's dot product
    if not np.shape(out.grad):  # it's a scalar
        func_self = np.multiply
        func_other = np.multiply
    else:
        func_self = partial(_dot_and_maybe_squeeze, self.shape)
        func_other = partial(_dot_and_maybe_squeeze, other.shape)

    if len(np.shape(out.grad)) == 1:
        # out.grad is a vector (in the numpy sense), but self_grad is a
        # matrix (b/c it's the "forced" transpose of a vector or a
        # matrix). We need to add an axis so that the dot product works
        # in numpy
        self.grad += func_self(out.grad[:, np.newaxis], self_grad)
    else:
        self.grad += func_self(out.grad, self_grad)
    other.grad += func_other(other_grad, out.grad)


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
    # is_elt_wise = False is for dot products
    def decorator(method):
        @wraps(method)
        def wrapper(self: Tensor, other: Tensor, *args, **kwargs):
            data, self_grad, other_grad = method(self, other, *args, **kwargs)
            out = Tensor(data)  # we need to reference this object for the chain_rule
            out._inputs = {self, other}

            def chain_rule():  # assume out.grad is set correctly
                if is_elt_wise:
                    self.grad += out.grad * self_grad
                    other.grad += out.grad * other_grad
                else:  # tricky
                    _dot_product_chain_rule(self, other, out, self_grad, other_grad)

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

    ####################################################################################
    ############################### ARITHMETIC FUNCTIONS ###############################
    ####################################################################################

    @_double_var()
    def __add__(self, other: Tensor) -> Tensor:
        data = self._data + other._data
        self_grad = 1
        other_grad = 1
        return data, self_grad, other_grad

    @_double_var()
    def __mul__(self, other: Tensor) -> Tensor:
        data = self._data * other._data

        def match_shape(self: Tensor, other: Tensor):
            if not other.shape:
                return np.ones_like(self) * other._data
            return other._data

        self_grad = match_shape(self, other)
        other_grad = match_shape(other, self)
        return data, self_grad, other_grad

    @_double_var(is_elt_wise=False)
    def dot(self, other: Tensor) -> Tensor:
        data = self._data @ other._data
        self_grad = _force_transpose(other._data)
        other_grad = _force_transpose(self._data)
        return data, self_grad, other_grad

    @_single_var
    def __pow__(self, exponent: float | int) -> Tensor:
        if not isinstance(exponent, (float, int)):
            raise TypeError(f"exponent must be a float/constant. Got {exponent}.")
        data = self._data**exponent
        grad = exponent * self._data ** (exponent - 1)
        return data, grad

    def sum(self, dim: int | None = None) -> Tensor:
        # TODO: add answer key explanation
        data = self._data.sum(axis=dim)
        out = Tensor(data)  # we need to reference this object for the chain_rule
        out._inputs = {self}

        def chain_rule():  # assume out.grad is set correctly
            if dim is None or (self.ndim <= 1 and dim == 0):
                # out.grad is a scalar
                grad = np.repeat(out.grad, repeats=self._data.size).reshape(
                    self._data.shape
                )
            elif dim == 0:
                # out.grad is a vector w/ shape (self.shape[1],) or (m,)
                grad = np.repeat(
                    out.grad[np.newaxis, :],  # make it 1 x m
                    repeats=self.shape[0],
                    axis=0,  # repeat n times to make it n x m
                )
            elif dim == 1:
                # out.grad is a vector w/ shape (self.shape[0],) or (n,)
                grad = np.repeat(
                    out.grad[:, np.newaxis],  # make it n x 1
                    repeats=self.shape[1],
                    axis=1,  # repeat m times to make it n x m
                )
            self.grad += grad

        out._backward = chain_rule
        return out

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

    ####################################################################################
    ################################# RE-SHAPE METHODS #################################
    ####################################################################################

    def mean(self, dim: int | None = None) -> Tensor:
        sum_ = self.sum(dim=dim)
        try:
            return sum_ / self.shape[dim]
        except IndexError:
            return sum_

    def __getitem__(self, key) -> Tensor:
        # TODO: add answer key explanation
        data = self._data[key]
        out = Tensor(data)  # we need to reference this object for the chain_rule
        out._inputs = {self}

        def chain_rule():  # assume out.grad is set correctly
            grad = np.zeros_like(self._data)
            grad[key] = out.grad
            self.grad += grad

        out._backward = chain_rule
        return out

    def take_along_dim(self, indices, dim: int | None) -> Tensor:
        # TODO: add answer key explanation
        data = np.take_along_axis(self._data, indices, axis=dim)
        out = Tensor(data)  # we need to reference this object for the chain_rule
        out._inputs = {self}

        def chain_rule():  # assume out.grad is set correctly
            grad = np.zeros_like(self._data)
            np.put_along_axis(grad, indices, values=out.grad, axis=dim)
            self.grad += grad

        out._backward = chain_rule
        return out

    @property
    def T(self) -> Tensor:
        # TODO: add answer key explanation
        data = self._data.T
        out = Tensor(data)  # we need to reference this object for the chain_rule
        out._inputs = {self}

        def chain_rule():  # assume out.grad is set correctly
            self.grad += out.grad.T

        out._backward = chain_rule
        return out

    def cat(self, other: Tensor, dim: int = -1) -> Tensor:
        raise NotImplementedError

    def stack(self, other: Tensor, dim: int = -1) -> Tensor:
        raise NotImplementedError

    def squeeze(self, dim: int = -1) -> Tensor:
        raise NotImplementedError

    def unsqueeze(self, dim: int = -1) -> Tensor:
        raise NotImplementedError

    ####################################################################################
    ############################### ACTIVATION FUNCTIONS ###############################
    ####################################################################################

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

    @_single_var
    def log_softmax(self, dim: int = -1) -> Tensor:
        raise NotImplementedError

    ####################################################################################
    ################################## LOSS FUNCTIONS ##################################
    ####################################################################################

    @_single_var
    def cross_entropy(self, labels: list[int], reduction: str = "mean") -> Tensor:
        # self._data are logits, i.e., unnormalized log-probabilities. labels are
        # sparsely encoded, so no soft labels
        # TODO: support observation weights
        if reduction not in {"sum", "mean"}:
            raise ValueError('reduction must be either "sum" or "mean".')

        # TODO: add answer key explanation
        y = np.array(labels)[:, np.newaxis]
        log_sum_exp = _log_sum_exp(self._data, dim=1, keepdims=True)
        data = (
            -np.take_along_axis(self._data, indices=y, axis=1).sum() + log_sum_exp.sum()
        )

        # data is the loss value. now for the gradient
        softmax = np.exp(self._data - log_sum_exp)
        is_label = np.zeros_like(self._data)
        np.put_along_axis(is_label, indices=y, values=1, axis=1)
        grad = softmax - is_label
        if reduction == "mean":
            denominator = len(labels)
            return data / denominator, grad / denominator
        return data, grad

    ####################################################################################
    ############################## CONVENIENCE FUNCTIONS ###############################
    ####################################################################################

    def __len__(self) -> int:
        return "wtf"

    @property
    def shape(self) -> tuple[int]:
        return np.shape(self._data)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def item(self) -> float:
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

    ####################################################################################
    ##################################### BACKWARD #####################################
    ####################################################################################

    def backward(self) -> None:
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
