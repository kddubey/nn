# nn
Neural network training and inference using only NumPy, vectorized. AKA
[micrograd](https://github.com/karpathy/micrograd) but w/ 2-D matrices.

<details>
<summary>Tips for others who want to re-implement backprop</summary>

- Watch this [extremely good video](https://www.youtube.com/watch?v=VMj-3S1tku0) (you
  can probably skip the last 40 min). I first learned backprop through math, and didn't
  really appreciate its elegance. That's b/c it's much more fruitfully thought about in
  terms of code: point to the object now and update it later.
- Take broadcasting for granted until you can't anymore.
- In case you're having trouble thinking about the gradient of the dot product of
  matrices/vectors, start with a vector-vector dot product, and then matrix-vector, and
  then matrix-matrix. Here's a sort of [answer
  key](http://cs231n.stanford.edu/vecDerivs.pdf).
- Just copy PyTorch's interface. That makes testing much easier.

</details>


## todo

- [ ] `requires_grad` / freezing functionality
- [ ] don't retain grads for non-leaf tensors
- [ ] arbitrary shapes
- [ ] basic NN framework
- [ ] tests
  - [ ] explicitly check that `(tensor - tensor.grad).shape == tensor.shape`
  - [ ] clever way to test `nn` code just by typing out `torch` code
