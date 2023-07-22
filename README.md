# nn
Neural network training and inference using only NumPy, vectorized. AKA
[micrograd](https://github.com/karpathy/micrograd) but w/ 2-D matrices.

<details>
<summary>Tips for others who want to re-implement backprop:</summary>

- Watch this [extremely good video](https://www.youtube.com/watch?v=VMj-3S1tku0) (you
  can probably skip the last 40 min). I first learned backprop through math, and didn't
  really appreciate its elegance. That's b/c it's much more fruitfully thought about in
  terms of code.
- In case you're having trouble thinking about the gradient of the dot product of
  matrices, start with a vector-vector dot product, and then matrix-vector, and then
  matrix-matrix. Here's a sort of [answer
  key](http://cs231n.stanford.edu/vecDerivs.pdf).
- When thinking about the gradient for elt-wise operations, constrain them to be the
  same shape as the data, i.e. the gradient `dY/dX` where `Y = elt_wise(X)` should have
  the same shape as `X`. Maybe this is obvious b/c we want to do `X -= lr * dY/dX`
  during gradient descent. But recall that Jacobians don't obey this property.
    - Sometimes you can be lazy and just let your array tool (NumPy) broadcast it. If
      you go down this route (I did), make sure to test carefully. (I realized just how
      much I take broadcasting for granted!)
- Just copy PyTorch's interface. That makes testing much easier.

</details>


## todo

- [ ] `requires_grad` / freezing functionality
- [ ] don't retain grads for non-leaf tensors
- [ ] arbitrary shapes
- [ ] basic NN framework
