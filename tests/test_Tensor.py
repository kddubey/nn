"""
Unit tests `nn.Tensor`
"""
from __future__ import annotations

import numpy as np

from nn import Tensor


def test_backward():
    # computational graph:
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

    # logistic regression
    w = Tensor([0, 1])
    x = Tensor([2, 3])
    logit = w.dot(x)
    logit.backward()
    assert np.all(w.grad == x._data)


def test___repr__():
    assert repr(Tensor([0]))
    assert repr(Tensor([0, 0]))
    assert repr(Tensor([[0, 0], [0, 0], [0, 0]]))
    assert repr(Tensor([[[0, 0], [0, 0]]]))
