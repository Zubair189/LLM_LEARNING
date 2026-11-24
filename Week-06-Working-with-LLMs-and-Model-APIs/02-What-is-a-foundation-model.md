# What is a foundation model?

> Week 06 – Working with LLMs and Model APIs · Question 2

> Note: Compare with [What is a pretrained model?](01-What-is-a-pretrained-model.md). "Pretrained" is about the training process; "foundation" is about scale, generality and being the base layer that many products are built on.

## Short Answer
A foundation model is a very large model pretrained on broad, diverse data (text, code, often images/audio) that can be adapted to a wide range of downstream tasks via prompting, tool use, or fine-tuning. The term was popularised by Stanford in 2021. Examples include Anthropic's Claude family, OpenAI's GPT models, Google's Gemini, and open-weight families like Llama, Qwen and Mistral.

## Detailed Explanation
Before foundation models, teams trained one model per task (a spam classifier, a translator, a summariser). A foundation model replaces many of these with **one general model + task-specific instructions**.

```
            ┌───────────── Foundation model ─────────────┐
            │  broad pretraining + instruction tuning    │
            └────────────────────────────────────────────┘
               │          │           │            │
           chatbot   summariser   code helper   RAG / agents
          (prompt)    (prompt)    (fine-tune)   (tools + docs)
```

Defining properties:
- **Scale** – billions of parameters, trained on enormous datasets.
- **Generality** – not built for one task; capabilities "emerge" across many.
- **Adaptability** – steered by prompts (system prompt, few-shot examples), tools, retrieval, or fine-tuning.
- **Often multimodal** – many current models accept images or documents as well as text.

| | Pretrained model (generic) | Foundation model |
|---|---|---|
| Size | Any (can be tiny) | Large |
| Scope | Could be one task | Many tasks |
| Example | DistilBERT sentiment classifier | Claude, GPT, Llama |
| Typical access | Download | Hosted API or open weights |

**Access modes:** closed models (e.g. `claude-sonnet-5-5`) are used through an inference API; open-weight models (e.g. Llama 3.2) can be run locally with Ollama or `transformers`.

## Example
**Analogy:** A foundation model is like the foundation of a building — one strong base on which you can construct a house, a shop or an office, just by changing what you build on top.

One model, three tasks, only the prompt changes:

```python
# pip install anthropic ; export ANTHROPIC_API_KEY=...
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

tasks = {
    "translate": "Translate to French: Good morning",
    "classify":  "Is this review positive or negative? 'Terrible battery life.' Answer with one word.",
    "code":      "Write a Python one-liner that reverses a string s.",
}
for name, prompt in tasks.items():
    resp = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    print(name, "->", resp.content[0].text.strip())
# Expected (wording may vary):
# translate -> Bonjour
# classify -> Negative
# code -> s[::-1]
```

## Common Mistakes / Misconceptions
- **Treating "foundation model" and "LLM" as identical.** Foundation models can be vision, audio or multimodal; LLMs are the text-focused subset.
- **Thinking bigger is always better.** Smaller models (e.g. `claude-haiku-5-5` or a local 3B model) are often cheaper, faster and good enough.
- **Assuming a foundation model is task-ready out of the box for regulated domains.** You still need evaluation, guardrails and often retrieval.
- **Confusing "open weights" with "open source".** Many open-weight models have restrictive licenses and undisclosed training data.

## Key Takeaways
- Foundation model = large, general, pretrained, adaptable base.
- One model serves many tasks via prompting, tools, RAG or fine-tuning.
- Accessed via hosted APIs (closed) or downloaded weights (open).
- Choose model size by task difficulty, latency and cost — not hype.

## Related Questions
- [What is a pretrained model?](01-What-is-a-pretrained-model.md)
- [What is the difference between local and cloud LLMs?](06-What-is-the-difference-between-local-and-cloud-LLMs.md)
- [What is an inference API?](07-What-is-an-inference-API.md)
- [What is few-shot prompting?](18-BONUS-What-is-few-shot-prompting.md)
