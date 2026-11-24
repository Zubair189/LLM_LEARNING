# Week 03 – Mathematics and Neural Network Foundations

This week builds the mathematical vocabulary every later topic depends on. Linear algebra (scalars, vectors, matrices, dot products) explains how data and weights are represented and why GPUs matter. Probability (distributions, softmax) explains what an LLM actually outputs: a distribution over the next token. The training loop (forward propagation, loss, backpropagation, gradient descent, learning rate) explains how a network goes from random weights to a useful model. By the end of the week you should be able to read `softmax(QK^T/√d)V` and a PyTorch training loop and know what every symbol and line means.

## Questions
1. [What is a scalar?](01-What-is-a-scalar.md)
2. [What is a vector?](02-What-is-a-vector.md)
3. [What is a matrix?](03-What-is-a-matrix.md)
4. [Why do neural networks use matrices?](04-Why-do-neural-networks-use-matrices.md)
5. [What is a dot product?](05-What-is-a-dot-product.md)
6. [What is probability?](06-What-is-probability.md)
7. [What is a probability distribution?](07-What-is-a-probability-distribution.md)
8. [What is an activation function?](08-What-is-an-activation-function.md)
9. [What is ReLU?](09-What-is-ReLU.md)
10. [What is softmax?](10-What-is-softmax.md)
11. [What is forward propagation?](11-What-is-forward-propagation.md)
12. [What is a loss function?](12-What-is-a-loss-function.md)
13. [What is gradient descent?](13-What-is-gradient-descent.md)
14. [What is backpropagation?](14-What-is-backpropagation.md)
15. [Why do neural networks require optimization?](15-Why-do-neural-networks-require-optimization.md)
16. **Bonus:** [What is a learning rate?](16-BONUS-What-is-a-learning-rate.md)

## Practical Exercise (1–2 hours): a 2-layer network in pure NumPy
Train a tiny network to solve XOR, a problem no linear model can solve.

1. **Data:** `X = [[0,0],[0,1],[1,0],[1,1]]`, `y = [0,1,1,0]`.
2. **Model:** `2 → 8 (ReLU) → 1 (sigmoid)`. Initialise weights with `rng.normal(scale=0.5)` and biases with zeros.
3. **Forward pass:** `z1 = X@W1+b1`, `h = relu(z1)`, `z2 = h@W2+b2`, `p = sigmoid(z2)`.
4. **Loss:** binary cross-entropy. Clip `p` to `[1e-7, 1-1e-7]` before taking the log.
5. **Backward pass** (derive these yourself first, then check): `dz2 = (p - y)/N`, `dW2 = h.T@dz2`, `dh = dz2@W2.T`, `dz1 = dh*(z1>0)`, `dW1 = X.T@dz1`.
6. **Update:** plain gradient descent for 2,000–5,000 steps. Print the loss every 500 steps.
7. **Experiments:**
   - Try learning rates 0.01, 0.1, 1.0 and 10, and describe what happens.
   - Remove the ReLU and show that the network can no longer solve XOR.
   - Add a finite-difference gradient check for one weight.
8. **Stretch:** replace the sigmoid output with a 2-class softmax plus cross-entropy, and confirm that the output gradient simplifies to `p - one_hot(y)`.

Success criterion: predictions round to `[0, 1, 1, 0]` and the loss falls below ~0.05.

## Explain Aloud (no notes)
1. **Why softmax + cross-entropy is how LLMs are trained.** Walk from logits, to probabilities, to the negative log-probability of the true next token, to perplexity.
2. **One full training step.** Cover the forward pass, loss, backpropagation (chain rule) and the optimiser update, and what the learning rate does. Make clear which part computes gradients and which part uses them.
3. **Why a neural network is "mostly matrix multiplication", and why that is not enough on its own.** Explain batching, GPUs, and why non-linear activations are required.
