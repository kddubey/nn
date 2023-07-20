"""
Unit tests `nn.Tensor`.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from nn import Tensor


@pytest.fixture(scope="module")
def atol() -> float:
    return 1e-07


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

    c = Tensor([2])
    d = Tensor([3])
    e = Tensor([4])
    a = c * d
    b = d * e
    da_root = a + b
    da_root.backward()

    assert np.all(c.grad == d._data)
    assert np.all(d.grad == (a.grad * c._data) + (b.grad * e._data))
    assert np.all(e.grad == d._data)
    assert np.all(a.grad == 1)
    assert np.all(b.grad == 1)
    assert np.all(da_root.grad == 1)


def test_backward_multi_variable(atol):
    # expected gradients from torch.Tensor
    X = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], requires_grad=True)
    Y = torch.tensor([[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]], requires_grad=True)
    W = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], requires_grad=True)
    Z = (2 * X - Y) * W / 3
    Z.sum().backward()

    # gradients from nn.Tensor
    X_ = Tensor(X.detach().numpy())
    Y_ = Tensor(Y.detach().numpy())
    W_ = Tensor(W.detach().numpy())
    Z_ = (2 * X_ - Y_) * W_ / 3
    Z_.backward()

    assert np.allclose(X.grad.numpy(), X_.grad, atol=atol)
    assert np.allclose(Y.grad.numpy(), Y_.grad, atol=atol)
    assert np.allclose(W.grad.numpy(), W_.grad, atol=atol)


def test_backward_nn(atol):
    # 1-hidden-layer binary regression. call this a hidden unit test ( ͡ ° ͜ʖ ͡ °)
    #         loss
    #           |
    #          MSE
    #          / \
    #         p   y
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

    # expected gradients from torch.Tensor
    y = torch.tensor([[0], [1]])
    X = torch.randn(size=(2, 3), requires_grad=True)
    W = torch.randn(size=(3, 2), requires_grad=True)
    v = torch.randn(size=(2, 1), requires_grad=True)
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

    # gradients from nn.Tensor
    y_ = Tensor(y.detach().numpy())
    X_ = Tensor(X.detach().numpy())
    W_ = Tensor(W.detach().numpy())
    v_ = Tensor(v.detach().numpy())
    Z_ = X_ @ W_
    U_ = Z_.relu()
    l_ = U_ @ v_
    o_ = l_.log_sigmoid()
    p_ = o_.exp()
    loss_ = (p_ - y_) ** 2
    loss_.backward()

    # test all gradients
    assert np.allclose(X.grad.numpy(), X_.grad, atol=atol)
    assert np.allclose(W.grad.numpy(), W_.grad, atol=atol)
    assert np.allclose(v.grad.numpy(), v_.grad, atol=atol)
    assert np.allclose(Z.grad.numpy(), Z_.grad, atol=atol)
    assert np.allclose(U.grad.numpy(), U_.grad, atol=atol)
    assert np.allclose(l.grad.numpy(), l_.grad, atol=atol)
    assert np.allclose(o.grad.numpy(), o_.grad, atol=atol)
    assert np.allclose(p.grad.numpy(), p_.grad, atol=atol)


def test___repr__():
    # for now just test that it runs
    assert repr(Tensor([0]))
    assert repr(Tensor([0, 0]))
    assert repr(Tensor([[0, 0], [0, 0], [0, 0]]))
    assert repr(Tensor([[[0, 0], [0, 0]]]))
