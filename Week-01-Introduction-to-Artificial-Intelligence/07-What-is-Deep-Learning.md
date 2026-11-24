# What is Deep Learning?

> Week 01 – Introduction to Artificial Intelligence · Question 7

## Short Answer
Deep learning is a subset of machine learning that uses **neural networks with many layers** ("deep" refers to the number of layers). It learns useful features directly from raw data such as pixels, audio or text, so humans don't have to design them by hand. Each layer builds more abstract representations from the layer below. Deep learning drives modern computer vision, speech recognition, and all of today's large language models.

## Detailed Explanation
**Classic ML vs. deep learning:**
```
Classic ML:     raw data ─► [human feature engineering] ─► simple model ─► output
Deep learning:  raw data ─► [layer 1 ─► layer 2 ─► ... ─► layer N] ─► output
                             (features are learned automatically)
```

**Hierarchical features (image example):**
- Layer 1 learns edges and colour blobs.
- Layer 2 combines them into corners and textures.
- Layer 3 builds parts: eyes, wheels.
- Deeper layers represent whole objects: "face", "car".

LLMs do something similar with text. Lower layers capture word pieces and syntax; higher layers capture meaning, facts and reasoning patterns.

| | Classic ML | Deep Learning |
|---|---|---|
| Features | Hand-crafted | Learned |
| Data needed | Hundreds to thousands of examples | Thousands to trillions |
| Compute | CPU is fine | GPUs/TPUs usually needed |
| Best on | Tabular / structured data | Images, audio, text, video |
| Interpretability | Often higher | Often lower |

**Why it took off around 2012:**
1. **Data:** the internet and large labelled datasets (ImageNet).
2. **Compute:** GPUs that do matrix maths quickly.
3. **Algorithms:** ReLU activations, better initialization, dropout, Adam, and later the Transformer (2017).

**Main architectures you'll meet:**
- **MLP**: stacked fully connected layers.
- **CNN**: convolutions, used for images.
- **RNN / LSTM**: sequences, now mostly replaced.
- **Transformer**: attention-based; the backbone of LLMs (Week 4).

## Example
**Analogy:** An assembly line. The first station sorts raw parts, the next builds sub-components, and later stations assemble the finished car. No single station "understands" the car, but together they produce it. Each layer of a deep network is one station.

**Python: a tiny 2-layer network solving XOR, which no single linear layer can do:**
```python
import math, random
random.seed(1)

X = [(0, 0), (0, 1), (1, 0), (1, 1)]
Y = [0, 1, 1, 0]
sig = lambda z: 1 / (1 + math.exp(-z))

H = 4  # hidden units
W1 = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(H)]
b1 = [0.0] * H
W2 = [random.uniform(-1, 1) for _ in range(H)]
b2, lr = 0.0, 0.5

for epoch in range(10000):
    for (x1, x2), y in zip(X, Y):
        h = [sig(W1[j][0]*x1 + W1[j][1]*x2 + b1[j]) for j in range(H)]  # layer 1
        out = sig(sum(W2[j]*h[j] for j in range(H)) + b2)                # layer 2
        d_out = out - y                                    # backprop (cross-entropy)
        for j in range(H):
            d_h = d_out * W2[j] * h[j] * (1 - h[j])
            W2[j] -= lr * d_out * h[j]
            W1[j][0] -= lr * d_h * x1
            W1[j][1] -= lr * d_h * x2
            b1[j] -= lr * d_h
        b2 -= lr * d_out

for (x1, x2) in X:
    h = [sig(W1[j][0]*x1 + W1[j][1]*x2 + b1[j]) for j in range(H)]
    print((x1, x2), round(sig(sum(W2[j]*h[j] for j in range(H)) + b2)))
# (0, 0) 0
# (0, 1) 1
# (1, 0) 1
# (1, 1) 0
```
The hidden layer learns an intermediate representation that makes XOR solvable. That is the essence of "depth". In practice you would use PyTorch or JAX, not hand-written loops.

## Common Mistakes / Misconceptions
- **"Deep learning is always better."** On small tabular datasets, gradient-boosted trees often win.
- **"Deep means it understands deeply."** "Deep" just means many layers.
- **"Deep learning is a separate field from ML."** It is a subset of ML.
- **Underestimating cost.** Training needs a lot of data, compute and energy. Most engineers *use* pretrained models rather than training from scratch.

## Key Takeaways
- Deep learning means multi-layer neural networks that learn features automatically.
- Layers build increasingly abstract representations.
- It succeeded because of data, GPUs and better algorithms; Transformers power LLMs.
- It is best for unstructured data (text, images, audio); classic ML often wins on small tabular data.

## Related Questions
- [What is a neural network?](./08-What-is-a-neural-network.md)
- [What are neurons, weights and biases?](./09-What-are-neurons-weights-and-biases.md)
- [What is backpropagation?](../Week-03-Mathematics-and-Neural-Network-Foundations/14-What-is-backpropagation.md)
- [What is a Transformer?](../Week-04-Transformer-Architecture/02-What-is-a-Transformer.md)
