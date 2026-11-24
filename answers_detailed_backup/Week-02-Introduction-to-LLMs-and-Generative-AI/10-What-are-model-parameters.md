# What are model parameters?

> Week 02 – Introduction to LLMs and Generative AI · Question 10

## Short Answer
Model parameters are the **learned numbers inside a model**: the weights and biases of its neural network, including the embedding tables and attention matrices. They are adjusted during training and fixed afterwards. Together they encode everything the model has learned. An LLM's size is usually quoted by its parameter count, such as "8B" or "70B" (billions). Parameter count roughly tracks capability and directly sets memory needs: at 16-bit precision each parameter takes 2 bytes, so a 70B model needs about 140 GB just to hold its weights.

## Detailed Explanation
**Parameters vs. hyperparameters vs. generation settings:**
| Term | What it is | Who sets it | Examples |
|---|---|---|---|
| **Parameters** | Learned weights and biases | The training algorithm | Billions of floats in the model files |
| **Hyperparameters** | Training/architecture choices | ML engineers | Learning rate, number of layers, batch size |
| **Inference (sampling) settings** | Per-request knobs | You, in the API call | `temperature`, `top_p`, `max_tokens` |

Confusingly, API docs call `temperature` and similar settings "parameters" too. In ML theory, "model parameters" means the learned weights.

**Where an LLM's parameters live (decoder-only Transformer):**
- **Token embedding matrix:** `vocab_size × d_model` (e.g. 128k × 4096 ≈ 0.5B).
- **Each Transformer layer:**
  - attention projections (Q, K, V, output);
  - feed-forward network (usually the largest share);
  - normalization weights.
- **Output (unembedding) layer**, sometimes shared with the embeddings.

**Memory rule of thumb for inference (weights only):**
| Precision | Bytes/param | 8B model | 70B model |
|---|---|---|---|
| FP32 | 4 | 32 GB | 280 GB |
| FP16 / BF16 | 2 | 16 GB | 140 GB |
| INT8 | 1 | 8 GB | 70 GB |
| 4-bit | 0.5 | 4 GB | 35 GB |

Add extra memory for the KV cache (which grows with context length and batch size) and for activations. Training needs **far** more memory (~16+ bytes/param with Adam optimizer states). That is why LoRA/QLoRA exist (Week 9).

**Bigger is not automatically better:**
- **Data quality and amount** matter as much as size; scaling laws balance parameters against training tokens.
- **Post-training** (instruction tuning, RL) strongly affects usefulness.
- **Mixture-of-Experts (MoE)** models have a large *total* parameter count, but only a subset is *active* per token. Their cost per token is closer to that of a smaller dense model.
- Proprietary providers often don't publish parameter counts at all.

> **Note on overlap:** Week 1's [What are neurons, weights and biases?](../Week-01-Introduction-to-Artificial-Intelligence/09-What-are-neurons-weights-and-biases.md) explains what a single weight *does* inside a neuron. This question is about parameters at LLM scale: counts, memory, and what "7B" means.

## Example
**Analogy:** Parameters are like the millions of tiny adjustments a pianist's muscles have learned over years of practice. You can't point to one muscle fibre that "knows" Chopin. The skill is spread across all of them, and once learned it is applied without re-learning.

**Python: count parameters and estimate memory for a Transformer-like model:**
```python
def transformer_params(vocab, d_model, n_layers, d_ff):
    embed = vocab * d_model
    attn = 4 * d_model * d_model                  # Q, K, V, O projections
    ffn = 2 * d_model * d_ff                      # up + down projection
    norms = 2 * d_model
    per_layer = attn + ffn + norms
    return embed + n_layers * per_layer           # output layer tied to embeddings

p = transformer_params(vocab=50_000, d_model=4096, n_layers=32, d_ff=16384)
print(f"{p/1e9:.2f}B parameters")                 # 6.65B parameters
for name, bytes_per in [("fp16", 2), ("int8", 1), ("4-bit", 0.5)]:
    print(f"{name:6s} ~{p * bytes_per / 1e9:.1f} GB")
# fp16   ~13.3 GB
# int8   ~6.6 GB
# 4-bit  ~3.3 GB
```
(This is simplified. Real models use variations such as grouped-query attention, gated feed-forward layers and untied embeddings.)

## Common Mistakes / Misconceptions
- **Mixing up parameters with tokens.** "70B parameters" is model size; "15T tokens" is the amount of training data.
- **Calling `temperature` a model parameter.** It is a sampling setting; the weights don't change.
- **"More parameters always means better answers."** A well-trained smaller model can beat an older, larger one.
- **Estimating GPU memory from weights alone.** The KV cache and runtime overhead add significantly, especially with long contexts.

## Key Takeaways
- Parameters are the learned weights and biases; they hold the model's knowledge.
- Size is quoted in parameters ("8B", "70B"); weight memory ≈ params × bytes per param.
- Hyperparameters and sampling settings are different things from learned parameters.
- Capability depends on parameters, data, training recipe and architecture (e.g. MoE), not size alone.

## Related Questions
- [What are neurons, weights and biases?](../Week-01-Introduction-to-Artificial-Intelligence/09-What-are-neurons-weights-and-biases.md)
- [What is the difference between an open-weight model and a proprietary model?](./13-What-is-the-difference-between-an-open-weight-model-and-a-proprietary-model.md)
- [What is quantization?](../Week-09-Fine-tuning-and-Model-Optimization/09-What-is-quantization.md)
- [What is the temperature parameter?](../Week-06-Working-with-LLMs-and-Model-APIs/12-What-is-the-temperature-parameter.md)
