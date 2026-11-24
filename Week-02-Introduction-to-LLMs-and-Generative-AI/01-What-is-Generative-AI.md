# What is Generative AI?

> Week 02 – Introduction to LLMs and Generative AI · Question 1

## Short Answer
Generative AI refers to models that **create new content**, such as text, images, audio, video or code, instead of only classifying or predicting a label. They learn the patterns (the probability distribution) of their training data and then **sample** new outputs that resemble it. Examples include LLMs like Claude, GPT and Gemini for text and code; diffusion models for images and video; and speech models for audio. The key contrast is with **discriminative** models, which answer "which category is this?" rather than "produce something new".

## Detailed Explanation
**Discriminative vs. generative:**
| | Discriminative AI | Generative AI |
|---|---|---|
| Learns | `P(label ∣ input)`, the boundary between classes | `P(data)` or `P(next piece ∣ previous pieces)`, how the data itself looks |
| Output | A label or number | New content |
| Example | "Is this email spam?" | "Write a reply to this email" |
| Typical models | Logistic regression, BERT classifiers | GPT/Claude-style LLMs, diffusion, VAEs, GANs |

**Main families of generative models:**
- **Autoregressive Transformers (LLMs):** generate one token at a time, each conditioned on everything before it. They are the standard for text and code, and are increasingly multimodal.
- **Diffusion models:** start from random noise and denoise it step by step into an image, video or audio clip.
- **GANs:** a generator competes against a discriminator. They were important historically and are now less common.
- **VAEs:** encode data into a latent space and decode samples from it. Often used inside diffusion pipelines.

**Why it took off in the 2020s:**
1. Transformers scaled well with data and compute.
2. Training on internet-scale data.
3. Instruction tuning and RLHF made the models *usable* as assistants (see the bonus question).
4. Cheap API access put them in reach of every developer.

**What LLM engineers build with it:** chat assistants, RAG (question answering over your documents), coding agents, summarization, data extraction, content generation, and multi-step AI agents that call tools.

**Limits to keep in mind:** outputs are *plausible*, not guaranteed to be *true* (hallucination). There are also open questions about copyright, bias, cost, latency and safety.

## Example
**Analogy:** A discriminative model is an art critic who can say "that's a Van Gogh". A generative model is an art student who has studied thousands of paintings and can paint a *new* picture in Van Gogh's style.

**Python: the smallest possible text generator, a character-level bigram model:**
```python
import random
from collections import defaultdict
random.seed(4)

corpus = "the cat sat on the mat. the cat ate the rat. the rat sat."
# learn P(next char | current char) by counting
counts = defaultdict(lambda: defaultdict(int))
for a, b in zip(corpus, corpus[1:]):
    counts[a][b] += 1

def generate(start="t", n=40):
    out = start
    for _ in range(n):
        nxt = counts[out[-1]]
        out += random.choices(list(nxt), weights=nxt.values())[0]  # sample
    return out

print(generate())   # the sat. mathe sat. sat. rat the rathe ra   <- new text, similar "style"
```
It learned a distribution from data and **sampled new text** from it. An LLM does the same thing with tokens instead of characters, billions of parameters, and a Transformer instead of a count table.

**Calling a hosted generative model (Anthropic SDK):**
```python
import anthropic
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
resp = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=200,
    messages=[{"role": "user", "content": "Write a two-line poem about rain."}],
)
print(resp.content[0].text)
```

## Common Mistakes / Misconceptions
- **"Generative AI = ChatGPT."** It also covers images, audio, video, code, 3D and molecules.
- **"It retrieves answers from a database."** It *generates* from learned patterns. That is why it can be fluent and still wrong.
- **"Generated means original and copyright-free."** Models can reproduce training data, and the legal questions are still being settled.
- **"Discriminative AI is obsolete."** Classifiers, rankers and embedding models are still essential, for example inside RAG and moderation.

## Key Takeaways
- Generative AI creates new content by learning and sampling from a data distribution.
- Discriminative models classify; generative models produce.
- LLMs (autoregressive Transformers) dominate text and code; diffusion dominates images and video.
- Outputs are plausible, not guaranteed correct, so engineering for reliability is the job.

## Related Questions
- [What is a Large Language Model (LLM)?](./02-What-is-a-Large-Language-Model-LLM.md)
- [What is a language model?](./05-What-is-a-language-model.md)
- [Why do LLMs sometimes hallucinate?](./12-Why-do-LLMs-sometimes-hallucinate.md)
- [What is Artificial Intelligence (AI)?](../Week-01-Introduction-to-Artificial-Intelligence/01-What-is-Artificial-Intelligence-AI.md)
