# What is a pretrained model?

> Week 06 – Working with LLMs and Model APIs · Question 1

> Note: "Pretrained" describes **how** a model was produced (trained in advance on a large dataset). "Foundation model" ([Q2](02-What-is-a-foundation-model.md)) describes a **role**: a very large, general pretrained model that many downstream applications build on. Every foundation model is pretrained; most pretrained models (e.g. a small sentiment classifier) are not foundation models.

## Short Answer
A pretrained model is a neural network whose weights have already been learned on a large dataset, so you can download or call it and use it immediately instead of training from scratch. For LLMs, pretraining means learning to predict the next token over trillions of tokens of text. You then use it as-is, prompt it, or fine-tune it on a smaller task-specific dataset (transfer learning).

## Detailed Explanation
Training a modern LLM from scratch costs millions of dollars of compute. Pretraining lets that cost be paid once and the result reused by everyone.

```
 Huge generic data ──► PRETRAINING ──► pretrained weights (base model)
                                             │
             ┌───────────────────────────────┼─────────────────────────┐
             ▼                               ▼                         ▼
     use via prompting           instruction/chat tuning       fine-tune on your data
                                  (RLHF, etc.)                 (e.g. legal Q&A)
```

Key ideas:
- **Base vs instruct models.** A *base* pretrained LLM just continues text. An *instruct/chat* model has had extra training (supervised fine-tuning, preference tuning) so it follows instructions. On Hugging Face you will see names like `Llama-3.2-1B` vs `Llama-3.2-1B-Instruct`.
- **Transfer learning.** General knowledge (grammar, facts, reasoning patterns) learned during pretraining transfers to new tasks, so fine-tuning needs far less data.
- **Artifacts.** A pretrained model is usually distributed as weights files (often `.safetensors` or GGUF) plus a config and tokenizer.

| Approach | Data needed | Compute | Typical use |
|---|---|---|---|
| Train from scratch | Billions–trillions of tokens | Massive | Model labs only |
| Use pretrained via prompt | None | Inference only | Most apps |
| Fine-tune pretrained | Hundreds–thousands of examples | Moderate | Domain style/format |

## Example
**Analogy:** Hiring a graduate who already speaks the language and knows general facts, then giving them a short onboarding for your company — rather than raising a child from birth to do the job.

```python
# pip install transformers torch
from transformers import pipeline

# Downloads pretrained weights from the Hugging Face Hub on first run
classifier = pipeline("sentiment-analysis",
                      model="distilbert-base-uncased-finetuned-sst-2-english")
print(classifier("The LLM answer was clear and helpful."))
# Expected (score varies slightly): [{'label': 'POSITIVE', 'score': 0.99...}]

generator = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
out = generator("Explain a pretrained model in one sentence:", max_new_tokens=40)
print(out[0]["generated_text"])
# Expected: the prompt followed by a short generated explanation
```

No training happened here — we only loaded someone else's learned weights and ran inference.

## Common Mistakes / Misconceptions
- **"Pretrained means it knows everything up to today."** It only knows its training data up to a cutoff date; use retrieval (RAG) for fresh facts.
- **Using a base model like a chatbot.** Base models ramble or continue your prompt; pick the `-Instruct`/chat variant for Q&A.
- **Assuming fine-tuning is always needed.** Prompting a strong pretrained model often suffices; fine-tune only when prompting plateaus.
- **Ignoring the license.** "Downloadable" does not mean "free for commercial use" — check the model card.

## Key Takeaways
- Pretrained = weights already learned on large data; ready to use.
- LLM pretraining is next-token prediction at huge scale.
- Base models complete text; instruct models follow instructions.
- Fine-tuning adapts a pretrained model cheaply via transfer learning.
- Always check the model card for cutoff date, intended use and license.

## Related Questions
- [What is a foundation model?](02-What-is-a-foundation-model.md)
- [What is Hugging Face?](03-What-is-Hugging-Face.md)
- [What is a model repository?](04-What-is-a-model-repository.md)
- [What is prompt engineering?](09-What-is-prompt-engineering.md)
