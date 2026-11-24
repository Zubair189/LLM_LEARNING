# What is a learning rate?

> Week 03 – Mathematics and Neural Network Foundations · Bonus question (added)

## Short Answer
The learning rate ($\eta$) is a hyperparameter that sets how big a step the optimiser takes in the direction of the negative gradient: $\theta \leftarrow \theta - \eta\nabla L$. If it is too small, training is painfully slow or gets stuck. If it is too large, the loss oscillates or diverges (often to NaN). It is usually the single most important hyperparameter to tune, and LLM training changes it over time with a schedule (warm-up then decay).

## Detailed Explanation
**Effect of different values** (on a simple bowl-shaped loss)
| Learning rate | Behaviour |
|---|---|
| Too small | tiny steps, very slow convergence, may never finish in the budget |
| Good | steady decrease, converges |
| Slightly too large | overshoots and oscillates around the minimum |
| Far too large | each step overshoots further, so the loss explodes (divergence, NaN) |

```
too small:  ●·●·●·●·●·●·●·→ ... (slow)
good:       ●───●──●─●●  (converges)
too large:  ●←──────────→●←──────────────→●  (diverges)
```

**Typical values**
- SGD: ~0.01–0.1
- Adam/AdamW for training Transformers from scratch: ~1e-4 to 6e-4 for mid-size models. Larger models generally use smaller peak learning rates.
- Fine-tuning: smaller (~1e-5 to 5e-5 for full fine-tuning). LoRA often uses ~1e-4 to 2e-4.

**Learning-rate schedules (standard for LLMs)**
1. **Warm-up:** ramp linearly from ~0 to the peak over the first few hundred to few thousand steps. This avoids instability while Adam's statistics and the weights are still settling.
2. **Decay:** cosine or linear decay to a small fraction of the peak, or "warmup-stable-decay" (constant, then a short final decay), which makes it easy to continue training.

```
lr
 |   /‾‾‾‾‾‾‾‾\___
 |  /             ‾‾‾\____
 | /                       ‾‾‾‾___
 +--------------------------------> steps
  warm-up      cosine decay
```

**Interactions**
- **Batch size:** larger batches give less noisy gradients and usually tolerate a larger learning rate.
- **Adaptive optimisers** (Adam) scale steps per parameter, but the global learning rate still sets the overall scale.
- **Gradient clipping** guards against occasional huge gradients, and works alongside the learning rate.

**How to choose:** start from published values for similar models, run a learning-rate range test (sweep from small to large and watch where the loss starts dropping fastest and where it blows up), then tune with short runs.

## Example
**Analogy:** Walking downhill in fog. Baby steps (small learning rate) are safe but take all day. Giant leaps (large learning rate) make you jump over the valley onto the opposite slope, possibly higher than where you started.

Minimise $L(w) = (w-3)^2$ from $w=0$ for 10 steps with different learning rates:

```python
def run(lr, steps=10):
    w = 0.0
    for _ in range(steps):
        w -= lr * 2 * (w - 3)       # gradient of (w-3)^2 is 2(w-3)
    return round(w, 4)

for lr in [0.01, 0.1, 0.5, 0.9, 1.1]:
    print(lr, run(lr))
# 0.01 0.5488    too small: barely moved toward 3
# 0.1 2.6779     good: steadily approaching 3
# 0.5 3.0        perfect for this particular bowl (one-step jump)
# 0.9 2.6779     overshoots back and forth, converging slowly
# 1.1 -15.5752   too large: diverges
```
Why: each step multiplies the distance from the minimum by $(1 - 2\eta)$. That factor shrinks the distance only if $|1-2\eta| < 1$, i.e. $0 < \eta < 1$.

## Common Mistakes / Misconceptions
- **Reusing a pre-training learning rate for fine-tuning.** That is usually far too high and can wreck the pretrained knowledge (catastrophic forgetting).
- **Skipping warm-up for Transformers.** Early instability or loss spikes are common without it.
- **Treating NaN losses as a code bug first.** A too-high learning rate is the most frequent cause.
- **Tuning other hyperparameters before the learning rate.** Fix the learning rate first. It dominates.

## Key Takeaways
- The learning rate scales every gradient step: $\theta \leftarrow \theta - \eta\nabla L$.
- Too small is slow. Too large oscillates or diverges.
- LLMs use warm-up followed by decay (cosine, linear or WSD).
- Fine-tuning uses much smaller learning rates than pre-training.

## Related Questions
- [What is gradient descent?](../Week-03-Mathematics-and-Neural-Network-Foundations/13-What-is-gradient-descent.md)
- [Why do neural networks require optimization?](../Week-03-Mathematics-and-Neural-Network-Foundations/15-Why-do-neural-networks-require-optimization.md)
- [What is a scalar?](../Week-03-Mathematics-and-Neural-Network-Foundations/01-What-is-a-scalar.md)
- [What is catastrophic forgetting?](../Week-09-Fine-tuning-and-Model-Optimization/13-What-is-catastrophic-forgetting.md)
