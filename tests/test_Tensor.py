"""
Unit tests `nn.Tensor`
"""
from __future__ import annotations

from nn import Tensor


def test_backward():
    c = Tensor(2)
    d = Tensor(3)
    e = Tensor(4)
    a = c * d
    b = d * e
    logit = a + b
    logit.backward()

    assert c.grad == d._data
    assert d.grad == (a.grad * c._data) + (b.grad * e._data)
    assert e.grad == d._data
    assert a.grad == 1
    assert b.grad == 1
    assert logit.grad == 1
