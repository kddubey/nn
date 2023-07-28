"""
Tests `nn.Tensor`. Most of these are integration tests. I've found that unit tests are
too easy to pass using specious code.

Here's how most of these tests work:
  1. Define a sequence of PyTorch operations which ends with a backward call,
     retaining gradients
  2. Copy over data from leaf torch.Tensors into nn.Tensors
  3. Repeat the same sequence of operations for the nn.Tensors
  4. Verify that all (leaf and non-leaf) gradients are equivalent or numerically close
     in order from root to leaf.

Maybe there's a way to automate this, but I'm just doing these manually for now.
Also I need to test that forward is correct too lol
I also should check that `(tensor._data - tensor.grad).shape == tensor.shape`.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import nn


@pytest.fixture(scope="module")
def atol() -> float:
    return 1e-06


def test__log_sum_exp(atol):
    X = torch.randn(size=(4, 3))
    lse_torch = torch.logsumexp(X, dim=1, keepdim=True)
    lse_nn = nn._tensor._log_sum_exp(X.numpy(), dim=1, keepdims=True)
    assert np.allclose(lse_torch, lse_nn, atol=atol)


def test_backward_single_variable():
    #        da_root
    #           |
    #           +
    #          / \
    #         a   b
    #         |   |
    #         *   *
    #        / \ / \
    #       c   d   e

    c = nn.Tensor([2])
    d = nn.Tensor([3])
    e = nn.Tensor([4])
    a = c * d
    b = d * e
    da_root = a + b
    da_root.backward()

    # easy enough to test conceptually and manually
    assert np.all(da_root.grad == 1)
    assert np.all(b.grad == 1)
    assert np.all(a.grad == 1)
    assert np.all(e.grad == d._data)
    assert np.all(d.grad == (a.grad * c._data) + (b.grad * e._data))
    assert np.all(c.grad == d._data)


def test_backward_multi_variable(atol):
    # torch.Tensor
    X = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], requires_grad=True)
    Y = torch.tensor([[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]], requires_grad=True)
    W = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], requires_grad=True)
    Z = (2 * X - Y) * W / 3
    Z.sum().backward()

    # nn.Tensor
    X_ = nn.Tensor(X.detach().numpy())
    Y_ = nn.Tensor(Y.detach().numpy())
    W_ = nn.Tensor(W.detach().numpy())
    Z_ = (2 * X_ - Y_) * W_ / 3
    Z_.backward()

    assert np.allclose(W.grad.numpy(), W_.grad, atol=atol)
    assert np.allclose(Y.grad.numpy(), Y_.grad, atol=atol)
    assert np.allclose(X.grad.numpy(), X_.grad, atol=atol)


@pytest.mark.parametrize(
    "size_and_dim",
    (
        ((1,), None),  # scalar
        ((3,), None),  # vector
        ((3,), 0),  # vector, same as dim=None
        ((3, 2), None),  # matrix, sum everything
        ((3, 2), 0),  # matrix, sum rows
        ((3, 2), 1),  # matrix, sum columns
    ),
)
def test_sum(size_and_dim: tuple[tuple[int], int], atol: float):
    size, dim = size_and_dim
    if dim is None or len(size) == 1:
        w_size = (1,)
    elif dim <= 1 and len(size) == 2:
        w_size = (size[1 - dim],)
    else:
        raise ValueError("not testing above 2-D shapes yet")

    # torch.Tensor
    X = torch.randn(size=size, requires_grad=True)
    w = torch.randn(size=w_size, requires_grad=True)
    x = X.sum(dim=dim)
    x.retain_grad()
    y = x * w
    y.retain_grad()
    z = y.sum()
    z.backward()

    # nn.Tensor
    X_ = nn.Tensor(X.detach().numpy())
    w_ = nn.Tensor(w.detach().numpy())
    x_ = X_.sum(dim=dim)
    y_ = x_ * w_
    z_ = y_.sum()
    z_.backward()

    assert np.allclose(y.grad.numpy(), y_.grad, atol)
    assert np.allclose(w.grad.numpy(), w_.grad, atol)
    assert np.allclose(x.grad.numpy(), x_.grad, atol)
    assert np.allclose(X.grad.numpy(), X_.grad, atol)


@pytest.mark.parametrize(
    "sizes",
    (
        ((1,), (1,)),  # scalar x scalar -> scalar
        ((2,), (2,)),  # vector x vector -> scalar
        ((3, 2), (2,)),  # matrix x vector -> vector
        ((2, 1), (1, 3)),  # vector x vector -> matrix
        ((2, 2), (2, 2)),  # matrix x matrix -> matrix (both square)
        ((3, 2), (2, 4)),  # matrix x matrix -> matrix (both non-square)
    ),
)
def test_dot(sizes, atol):
    size1, size2 = sizes

    # torch.Tensor
    X = torch.randn(size=size1, requires_grad=True)
    Y = torch.randn(size=size2, requires_grad=True)
    Z = X @ Y
    Z.retain_grad()
    U = Z.sum()
    U.backward()

    # nn.Tensor
    X_ = nn.Tensor(X.detach().numpy())
    Y_ = nn.Tensor(Y.detach().numpy())
    Z_ = X_ @ Y_
    U_ = Z_.sum()
    U_.backward()

    assert np.allclose(Z.grad.numpy(), Z_.grad, atol=atol)
    assert np.allclose(Y.grad.numpy(), Y_.grad, atol=atol)
    assert np.allclose(X.grad.numpy(), X_.grad, atol=atol)


def test_backward_nn(atol):
    # 1-hidden-layer binary regression. call it a hidden unit test ( ͡ ° ͜ʖ ͡ °)
    #         loss
    #           |
    #          MSE
    #          / \
    #         p   y
    #         |
    #        exp
    #         |
    #         o
    #         |
    #     log_sigmoid
    #         |
    #         l
    #         |
    #         @
    #        / \
    #       U   v
    #       |
    #      relu
    #       |
    #       Z
    #       |
    #       @
    #      / \
    #     X   W

    # torch.Tensor
    y = torch.tensor([0, 1])
    X = torch.randn(size=(2, 3), requires_grad=True)
    W = torch.randn(size=(3, 2), requires_grad=True)
    v = torch.randn(size=(2,), requires_grad=True)
    Z = X @ W
    Z.retain_grad()
    U = torch.relu(Z)
    U.retain_grad()
    l = U @ v
    l.retain_grad()
    o: torch.Tensor = torch.nn.functional.logsigmoid(l)
    o.retain_grad()
    p = torch.exp(o)
    p.retain_grad()
    loss = ((p - y) ** 2).sum()
    loss.backward()

    # nn.Tensor
    y_ = nn.Tensor(y.detach().numpy())
    X_ = nn.Tensor(X.detach().numpy())
    W_ = nn.Tensor(W.detach().numpy())
    v_ = nn.Tensor(v.detach().numpy())
    Z_ = X_ @ W_
    U_ = Z_.relu()
    l_ = U_ @ v_
    o_ = l_.log_sigmoid()
    p_ = o_.exp()
    loss_ = ((p_ - y_) ** 2).sum()
    loss_.backward()

    # test all gradients
    assert np.allclose(p.grad.numpy(), p_.grad, atol=atol)
    assert np.allclose(o.grad.numpy(), o_.grad, atol=atol)
    assert np.allclose(l.grad.numpy(), l_.grad, atol=atol)
    assert np.allclose(U.grad.numpy(), U_.grad, atol=atol)
    assert np.allclose(Z.grad.numpy(), Z_.grad, atol=atol)
    assert np.allclose(v.grad.numpy(), v_.grad, atol=atol)
    assert np.allclose(W.grad.numpy(), W_.grad, atol=atol)
    assert np.allclose(X.grad.numpy(), X_.grad, atol=atol)


def test___getitem__(atol):
    # torch.Tensor
    X = torch.randn(size=(2, 3), requires_grad=True)
    z = torch.randn(size=(2,), requires_grad=True)
    x = X[0, 0:2]
    x.retain_grad()
    y = x @ z
    y.retain_grad()
    y.backward()

    # nn.Tensor
    X_ = nn.Tensor(X.detach().numpy())
    z_ = nn.Tensor(z.detach().numpy())
    x_ = X_[0, 0:2]
    y_ = x_ @ z_
    y_.backward()

    assert np.allclose(z.grad.numpy(), z_.grad, atol=atol)
    assert np.allclose(x.grad.numpy(), x_.grad, atol=atol)
    assert np.allclose(X.grad.numpy(), X_.grad, atol=atol)


def test_take_along_dim(atol):
    # torch.Tensor
    A = torch.randn(size=(2, 3), requires_grad=True)
    B = torch.randn(size=(2, 4), requires_grad=True)
    indices = torch.tensor([[1, 2], [0, 1]])
    C = torch.take_along_dim(A, indices, dim=-1)
    C.retain_grad()
    D = C @ B
    D.retain_grad()
    d = D.sum()
    d.backward()

    # nn.Tensor
    A_ = nn.Tensor(A.detach().numpy())
    B_ = nn.Tensor(B.detach().numpy())
    indices_ = indices.numpy()
    C_ = A_.take_along_dim(indices_, dim=-1)
    D_ = C_ @ B_
    d_ = D_.sum()
    d_.backward()

    assert np.allclose(D.grad.numpy(), D_.grad, atol=atol)
    assert np.allclose(C.grad.numpy(), C_.grad, atol=atol)
    assert np.allclose(B.grad.numpy(), B_.grad, atol=atol)
    assert np.allclose(A.grad.numpy(), A_.grad, atol=atol)


def test_T(atol):
    # torch.Tensor
    X = torch.randn(size=(2, 3), requires_grad=True)
    W = torch.randn(size=(2, 3), requires_grad=True)
    Y = X.T
    Y.retain_grad()
    Z = Y @ W
    Z.sum().backward()

    # nn.Tensor
    X_ = nn.Tensor(X.detach().numpy())
    W_ = nn.Tensor(W.detach().numpy())
    Y_ = X_.T
    Z_ = Y_ @ W_
    Z_.sum().backward()

    assert np.allclose(Y.grad.numpy(), Y_.grad, atol)
    assert np.allclose(X.grad.numpy(), X_.grad, atol)
    assert np.allclose(W.grad.numpy(), W_.grad, atol)


def test_log_softmax(atol=1e-04):  # seems quite flaky
    # torch.Tensor
    X = torch.randn(size=(2, 2), requires_grad=True)
    W = torch.randn(size=(2, 3), requires_grad=True)
    L = X @ W
    W.retain_grad()
    X.retain_grad()
    L.retain_grad()
    l = L.log_softmax(dim=1)
    l.sum().backward()

    # nn.Tensor
    X_ = nn.Tensor(X.detach().numpy())
    W_ = nn.Tensor(W.detach().numpy())
    L_ = X_ @ W_
    l_ = L_.log_softmax()
    l_.sum().backward()

    assert np.allclose(L.grad.numpy(), L_.grad, atol)
    assert np.allclose(W.grad.numpy(), W_.grad, atol)
    assert np.allclose(X.grad.numpy(), X_.grad, atol)


def test_nll_loss(atol=1e-04):
    # torch.Tensor
    X = torch.randn(size=(2, 2), requires_grad=True)
    W = torch.randn(size=(2, 3), requires_grad=True)
    y = torch.randint(low=0, high=W.shape[1], size=(X.shape[0],))
    L = X @ W
    E = L.log_softmax(dim=1)
    W.retain_grad()
    X.retain_grad()
    L.retain_grad()
    E.retain_grad()
    l = torch.nn.functional.nll_loss(E, y, reduction="mean")
    l.backward()

    # nn.Tensor
    X_ = nn.Tensor(X.detach().numpy())
    W_ = nn.Tensor(W.detach().numpy())
    y_ = y.detach().numpy()
    L_ = X_ @ W_
    E_ = L_.log_softmax()
    l_ = E_.nll_loss(y_, reduction="mean")
    l_.backward()

    assert np.allclose(E.grad.numpy(), E_.grad, atol)
    assert np.allclose(L.grad.numpy(), L_.grad, atol)
    assert np.allclose(W.grad.numpy(), W_.grad, atol)
    assert np.allclose(X.grad.numpy(), X_.grad, atol)


def test_cross_entropy(atol):
    # torch.Tensor
    X = torch.randn(size=(2, 2), requires_grad=True)
    W = torch.randn(size=(2, 3), requires_grad=True)
    y = torch.randint(low=0, high=W.shape[1], size=(X.shape[0],))
    L = X @ W
    W.retain_grad()
    X.retain_grad()
    L.retain_grad()
    l = torch.nn.functional.cross_entropy(L, y, reduction="mean")
    l.backward()

    # nn.Tensor
    X_ = nn.Tensor(X.detach().numpy())
    W_ = nn.Tensor(W.detach().numpy())
    y_ = y.detach().numpy()
    L_ = X_ @ W_
    l_ = L_.cross_entropy(y_, reduction="mean")
    l_.backward()

    assert np.allclose(L.grad.numpy(), L_.grad, atol)
    assert np.allclose(W.grad.numpy(), W_.grad, atol)
    assert np.allclose(X.grad.numpy(), X_.grad, atol)


@pytest.mark.parametrize("shape", (1, 2, (2, 3), (2, 3, 4)))
def test___repr__(shape):
    # I have a (bad?) idea. If I change the name of the class from "Tensor" to "array",
    # then, b/c of the way Tensor's repr works, it should be exactly the same as numpy's
    # pretty repr. So let's do that for a few different shapes
    nn.Tensor.__name__ = "array"
    numpy_array = np.zeros(shape)
    assert repr(nn.Tensor(numpy_array)) == repr(numpy_array)
