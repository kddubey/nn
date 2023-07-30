# nn
Neural network training and inference using only NumPy, vectorized.
  * AKA [micrograd](https://github.com/karpathy/micrograd) but w/ 2-D matrices
  * AKA a transparent (but probably not accurate) implementation of PyTorch's `Tensor`,
    assuming broadcasting and dot products are given operations. 

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
  gradient to the right place / you're just directing traffic
- Just copy PyTorch's interface. That makes testing much easier.

</details>

## Usage

Here's an NN for classification with 1 hidden layer:

```python
import numpy as np
import nn

# input data parameters
num_observations = 100
input_dim = 10
num_classes = 3
rng_seed = abs(hash("waddup"))

# simulate input data
rng = np.random.default_rng(rng_seed)
y = nn.Tensor(rng.integers(0, num_classes, size=num_observations))
X = nn.Tensor(rng.normal(size=(num_observations, input_dim)))

# weights
hidden_size = 20
W1 = nn.Tensor(rng.normal(size=(input_dim, hidden_size)))
W2 = nn.Tensor(rng.normal(size=(hidden_size, num_classes)))

# forward pass
H1 = X @ W1
H1_relu = H1.relu()
H2 = H1_relu @ W2
loss = H2.cross_entropy(y)

# backward pass
loss.backward()
```


## Installation

```
python -m pip install git+https://github.com/kddubey/nn.git
```


## Todo

- [ ] actually support tensors
- [ ] tests
  - [ ] explicitly check that `(tensor._data - tensor.grad).shape == tensor.shape`
  - [ ] clever way to test `nn` code just by typing out `torch` code
- [ ] basic NN framework
- [ ] `requires_grad` / freezing functionality
- [ ] don't retain grads for non-leaf tensors
