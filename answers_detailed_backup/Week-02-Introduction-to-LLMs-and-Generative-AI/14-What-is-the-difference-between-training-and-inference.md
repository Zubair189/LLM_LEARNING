# What is the difference between training and inference?

> Week 02 – Introduction to LLMs and Generative AI · Question 14

## Short Answer
**Training** is the phase in which a model *learns*. It processes large amounts of data, measures its error with a loss function, and updates its weights through backpropagation and gradient descent. **Inference** is the phase in which the trained model is *used*. Its weights are frozen, and it only runs forward passes to produce predictions or generated text for new inputs. Training happens rarely, is extremely compute-intensive, and is done by model builders. Inference happens on every user request, must be fast and cheap, and is what most LLM engineers work with day to day.

## Detailed Explanation
```
TRAINING (learn)                                   INFERENCE (use)
data ─► forward pass ─► loss ─► backward pass       prompt ─► forward pass ─► output
          ▲                       │                            (weights frozen)
          └──── update weights ◄──┘   repeat billions of times
```

| Dimension | Training | Inference |
|---|---|---|
| Goal | Learn parameters | Produce outputs |
| Weights | Updated every step | Fixed (read-only) |
| Computation | Forward **+ backward** pass + optimizer step | Forward pass only |
| Memory per parameter | ~16+ bytes (weights, gradients, Adam states, mixed precision) | ~0.5–2 bytes (weights) + KV cache |
| Data | Huge datasets (trillions of tokens for pretraining) | One request at a time (batched for efficiency) |
| Frequency | Once per model version (weeks/months for pretraining) | Millions of times a day |
| Hardware | Thousands of GPUs/TPUs in tight clusters | One GPU to small clusters; sometimes a laptop or phone |
| Optimized for | Throughput and stability of learning | Latency, throughput, cost per token |
| Who does it | Labs / ML teams (or you, when fine-tuning) | Every app that calls the model |

**Where fine-tuning fits:** fine-tuning (Week 9) *is* training. It is just smaller, starting from pretrained weights and using your own data. Techniques like LoRA cut its memory cost.

**"In-context learning" is not training.** Giving examples in a prompt changes the model's output for that request only. No weights change, so it is still inference.

**Cost picture:** a frontier pretraining run costs a very large amount once, while inference cost grows with every user. For a popular model, total inference spend can exceed training spend. This is why efficient inference (quantization, caching, batching, smaller models) matters so much.

> **Note on overlap:**
> - [What is inference?](./11-What-is-inference.md) explains *how* inference works internally (prefill/decode, KV cache, sampling). This question is the *side-by-side comparison* of the two phases.
> - Don't confuse this with Week 1's [training vs testing data](../Week-01-Introduction-to-Artificial-Intelligence/11-What-is-the-difference-between-training-and-testing-data.md). That one is about **splitting datasets** to evaluate generalization, not about lifecycle phases. (Running a model on the test set *is* inference, but its purpose is evaluation.)

## Example
**Analogy:** Training is going to medical school: years of study, exams, and correcting mistakes. Inference is the doctor seeing a patient: applying what they learned in minutes, without going back to school for each patient.

**Python: the same tiny model in training mode vs. inference mode:**
```python
# Model: y = w * x   (learn w from data where the true w = 3)
data = [(1, 3), (2, 6), (3, 9), (4, 12)]
w, lr = 0.0, 0.01

# ---- TRAINING: forward + loss + gradient + update ----
for epoch in range(200):
    for x, y in data:
        pred = w * x                 # forward
        grad = 2 * (pred - y) * x    # d(loss)/dw for squared error
        w -= lr * grad               # update weights
print("learned w:", round(w, 3))     # learned w: 3.0

# ---- INFERENCE: forward only, weights frozen ----
def infer(x, weight=w):
    return weight * x
print("prediction for x=10:", round(infer(10), 2))   # prediction for x=10: 30.0
```
Training changed `w` hundreds of times. Inference only *reads* it.

## Common Mistakes / Misconceptions
- **"The model keeps learning from user chats in real time."** Deployed weights are frozen. Any learning from feedback happens in a *later*, separate training run.
- **"Fine-tuning is a kind of inference" or "prompting is a kind of training."** Fine-tuning changes weights (training); prompting doesn't (inference).
- **Sizing inference hardware with training numbers, or the reverse.** Their memory needs differ by roughly an order of magnitude.
- **Ignoring inference cost** because "training was the expensive part". At scale it often isn't.

## Key Takeaways
- Training learns weights (forward + backward + update); inference uses frozen weights (forward only).
- Training is rare, huge and costly; inference is constant and latency-sensitive.
- Fine-tuning is training; in-context learning and prompting are inference.
- LLM application engineers mostly optimize inference: cost, latency and quality.

## Related Questions
- [What is inference?](./11-What-is-inference.md)
- [What is the difference between training and testing data?](../Week-01-Introduction-to-Artificial-Intelligence/11-What-is-the-difference-between-training-and-testing-data.md)
- [What is backpropagation?](../Week-03-Mathematics-and-Neural-Network-Foundations/14-What-is-backpropagation.md)
- [What is fine-tuning?](../Week-09-Fine-tuning-and-Model-Optimization/02-What-is-fine-tuning.md)
