# nn
Neural network training and inference using only NumPy, vectorized.
  * AKA [micrograd](https://github.com/karpathy/micrograd) but w/ 2-D matrices
  * AKA a transparent implementation of PyTorch's `Tensor`, assuming broadcasting and
    dot products are given operations. 

<details>
<summary>Tips for others who want to re-implement backprop</summary>

- Closely follow this [extremely good
  video](https://www.youtube.com/watch?v=VMj-3S1tku0) (you can probably skip the last 40
  min). I first learned backprop through math, and didn't really appreciate its
  elegance. That's b/c it's much more fruitfully thought about in terms of code: point
  to the object now, and update it later when you know the one other thing you need to
  know.
- Take broadcasting for granted until you can't anymore.
  - Because of this project, I moved broadcasting up to #1 in my top 5 algorithms I take
    for granted.
- If you're having trouble thinking about the gradient of the dot product of
  matrices/vectors, start with a vector-vector dot product, and then matrix-vector, and
  then matrix-matrix. Here's a sort of [answer
  key](http://cs231n.stanford.edu/vecDerivs.pdf). I'll write a different one describing
  how I thought about condensing derivatives into the right vector/matrix.
    - Maybe I'm doing something wrong, but I found that I had to fight numpy's dot
      product a bit. It treats 1-D vectors pretty strictly.
- For slicing/re-shape operations, just directly code the output of the chain rule
  instead of relying on correct multiplication operations. It's computationally a bit
  better, and feels easier to code and think through: you're just passing on the
  gradient!
- Just copy PyTorch's interface. That makes testing much easier.

</details>


## todo

- [ ] `requires_grad` / freezing functionality
- [ ] don't retain grads for non-leaf tensors
- [ ] arbitrary shapes
- [ ] basic NN framework
- [ ] tests
  - [ ] explicitly check that `(tensor._data - tensor.grad).shape == tensor.shape`
  - [ ] clever way to test `nn` code just by typing out `torch` code
