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
    return 1e-06


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


def test_backward_multi_variable():
    # expected gradients from torch.Tensor
    X = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], requires_grad=True)
    Y = torch.tensor([[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]], requires_grad=True)
    W = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], requires_grad=True)
    Z = (X + Y) * W
    Z.sum().backward()

    # observed gradients from nn.Tensor
    X_ = Tensor(X.detach().numpy())
    Y_ = Tensor(Y.detach().numpy())
    W_ = Tensor(W.detach().numpy())
    Z_ = (X_ + Y_) * W_
    Z_.backward()

    assert np.all(X.grad.numpy() == X_.grad)
    assert np.all(Y.grad.numpy() == Y_.grad)
    assert np.all(W.grad.numpy() == W_.grad)


def test_backward_nn(atol):
    # 1-hidden-layer binary classification. computational graph:
    #         loss
    #           |
    #          NLL
    #          / \
    #         y   p
    #             |
    #          sigmoid
    #             |
    #             l
    #             |
    #             @
    #            / \
    #           V   U
    #               |
    #              relu
    #               |
    #               Z
    #               |
    #               @
    #              / \
    #             X   W

    # expected gradients from torch.Tensor
    # y = torch.tensor([[0], [1]])
    X = torch.randn(size=(2, 3), requires_grad=True)
    W = torch.randn(size=(3, 2), requires_grad=True)
    V = torch.randn(size=(2, 1), requires_grad=True)
    Z = X @ W
    Z.retain_grad()
    U = torch.relu(Z)
    U.retain_grad()
    l = U @ V
    l.retain_grad()
    p = torch.sigmoid(l)
    p.sum().backward()

    # observed gradients from nn.Tensor
    # y_ = Tensor(y.detach().numpy())
    X_ = Tensor(X.detach().numpy())
    W_ = Tensor(W.detach().numpy())
    V_ = Tensor(V.detach().numpy())
    Z_ = X_.dot(W_)
    U_ = Z_.relu()
    l_ = U_.dot(V_)
    p_ = l_.sigmoid()
    p_.backward()

    # test all gradients except root
    assert np.allclose(X.grad.numpy(), X_.grad, atol=atol)
    assert np.allclose(W.grad.numpy(), W_.grad, atol=atol)
    assert np.allclose(V.grad.numpy(), V_.grad, atol=atol)
    assert np.allclose(Z.grad.numpy(), Z_.grad, atol=atol)
    assert np.allclose(U.grad.numpy(), U_.grad, atol=atol)
    assert np.allclose(l.grad.numpy(), l_.grad, atol=atol)


def test___repr__():
    # for now just test that it runs
    assert repr(Tensor([0]))
    assert repr(Tensor([0, 0]))
    assert repr(Tensor([[0, 0], [0, 0], [0, 0]]))
    assert repr(Tensor([[[0, 0], [0, 0]]]))
