# What is inference?

> Week 02 – Introduction to LLMs and Generative AI · Question 11

## Short Answer
Inference means **using a trained model to make predictions or generate output** on new input. The weights are frozen; nothing is learned. For an LLM, inference means taking your prompt, running it through the network, and producing the response **one token at a time**. Each new token is fed back in to predict the next one, until the model emits a stop token or reaches `max_tokens`. Every API call or chat message is an inference request. In production, inference **latency**, **throughput** and **cost per token** are what engineers optimize.

## Detailed Explanation
**The two phases of LLM inference:**
```
Prompt: "The sky is"

PREFILL (once)                         DECODE (loop, one token per step)
[The][sky][is] ──all in parallel──►    predict ─► sample " blue" ─► append
  builds the KV cache                  predict ─► sample "."     ─► append
  ⇒ determines time-to-first-token     predict ─► sample <eos>   ─► stop
                                       ⇒ determines tokens/second
```
- **Prefill:** reads the whole prompt in one parallel pass. Its speed mainly determines **time to first token (TTFT)**.
- **Decode:** generates output tokens one by one, reusing a **KV cache** so earlier tokens are not recomputed. Its speed determines **tokens per second**.

**Sampling at each step:** the model outputs a probability for every vocabulary token. A decoding strategy chooses one:
- **Greedy:** always pick the most likely token.
- **Temperature / top-p / top-k sampling:** add controlled randomness (Week 6).

**Key inference metrics:**
| Metric | Meaning |
|---|---|
| TTFT | Time until the first token streams back |
| Output tokens/sec | Generation speed |
| End-to-end latency | TTFT + output_tokens / speed |
| Throughput | Total tokens/sec across all users on the hardware |
| Cost | $ per million input/output tokens (API) or GPU-hours (self-hosted) |

**Where inference runs:**
- **Hosted APIs** (Anthropic, OpenAI, Google, ...): you send HTTP requests, and the provider runs the GPUs.
- **Self-hosted:** open-weight models served with engines such as vLLM, SGLang, TGI or llama.cpp/Ollama.

**Optimization techniques (later weeks):** quantization, batching, KV-cache reuse, prompt caching, speculative decoding, smaller or distilled models, and streaming for *perceived* speed.

> **Note on overlap:** Q14 ([training vs inference](./14-What-is-the-difference-between-training-and-inference.md)) *contrasts* the two lifecycle phases. This question explains *how inference itself works*. Week 6's [What is an inference API?](../Week-06-Working-with-LLMs-and-Model-APIs/07-What-is-an-inference-API.md) covers calling inference as a hosted web service.

## Example
**Analogy:** Training is a chef spending years learning to cook. Inference is the chef cooking your order tonight using those skills, quickly and without re-learning anything.

**Python: autoregressive decoding loop with a toy "model":**
```python
import math

# Toy model: next-token logits depend only on the last token
LOGITS = {
    "The": {"sky": 2.0, "cat": 1.0},
    "sky": {"is": 3.0},
    "is":  {"blue": 2.5, "green": 0.5},
    "blue": {"<eos>": 3.0},
}

def softmax(d):
    m = max(d.values()); e = {k: math.exp(v - m) for k, v in d.items()}
    s = sum(e.values()); return {k: v / s for k, v in e.items()}

tokens, max_tokens = ["The"], 10
for _ in range(max_tokens):                 # decode loop
    probs = softmax(LOGITS[tokens[-1]])     # "forward pass"
    nxt = max(probs, key=probs.get)         # greedy decoding
    if nxt == "<eos>":
        break
    tokens.append(nxt)
print(" ".join(tokens))  # The sky is blue
```

**A real inference call with streaming (Anthropic SDK):**
```python
import anthropic
client = anthropic.Anthropic()
with client.messages.stream(
    model="claude-sonnet-5-5",
    max_tokens=100,
    messages=[{"role": "user", "content": "Why is the sky blue? One sentence."}],
) as stream:
    for text in stream.text_stream:     # tokens arrive as they are decoded
        print(text, end="", flush=True)
```

## Common Mistakes / Misconceptions
- **"The model learns from my prompts during inference."** Weights don't change. In-context learning is temporary and limited to that request.
- **"Inference is cheap because training was the expensive part."** At scale, total inference spend often *exceeds* training cost.
- **"Long outputs cost the same time as long prompts."** Output tokens are generated one at a time and dominate latency.
- **"Same prompt, same answer."** With sampling (temperature > 0) outputs vary, and even temperature 0 isn't perfectly deterministic on every platform.

## Key Takeaways
- Inference = running a frozen trained model on new inputs to get outputs.
- LLM inference has two phases: parallel prefill (sets TTFT), then token-by-token decode.
- The KV cache, batching and quantization make inference efficient.
- Optimize for TTFT, tokens/sec, throughput and cost; stream responses for better UX.

## Related Questions
- [What is the difference between training and inference?](./14-What-is-the-difference-between-training-and-inference.md)
- [What is an inference API?](../Week-06-Working-with-LLMs-and-Model-APIs/07-What-is-an-inference-API.md)
- [What is response streaming?](../Week-06-Working-with-LLMs-and-Model-APIs/14-What-is-response-streaming.md)
- [What are top-k and top-p sampling?](../Week-06-Working-with-LLMs-and-Model-APIs/13-What-are-top-k-and-top-p-sampling.md)
