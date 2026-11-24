# Compare RAG and fine-tuning.

> Week 12 – Final Project and Revision · Question 6

## Short Answer
**RAG** changes *what the model sees* at inference time by retrieving relevant documents into the prompt; it is best for knowledge that is large, private, or frequently changing, and it gives citations. **Fine-tuning** changes *the model's weights* by training on examples; it is best for teaching consistent behavior, style, format, or a narrow skill. Rule of thumb: **RAG for knowledge, fine-tuning for behavior** — and many production systems combine them after first exhausting good prompting.

## Detailed Explanation

### Side-by-side architecture

```
 RAG (knowledge at runtime)                 FINE-TUNING (behavior in weights)
 ─────────────────────────                  ──────────────────────────────────
 Docs ─▶ chunk ─▶ embed ─▶ Vector DB        Examples (prompt → ideal answer)
                               │                    │
 Question ─▶ retrieve top-k ◀──┘                    ▼  SFT / LoRA / QLoRA
        │                                    Base model ─▶ Tuned model (new weights)
        ▼                                                     │
 Prompt = instructions + chunks + question   Question ───────▶│
        ▼                                                     ▼
   Base LLM (unchanged) ─▶ answer + cites    Answer in learned style/format

 Update knowledge: re-index docs (minutes)  Update behavior: retrain (hours–days)
```

### Comparison table

| Dimension | RAG | Fine-tuning |
|---|---|---|
| Changes | Prompt/context | Model weights |
| Best for | Facts, policies, docs, fresh data | Tone, format, domain jargon, classification, narrow tasks |
| Freshness | Update the index anytime | Stale until retrained |
| Citations / traceability | Natural (chunk sources) | Hard — knowledge is "baked in" |
| Data needed | Documents (no labels) | Curated input→output examples (hundreds–thousands) |
| Upfront cost | Low–medium (pipeline, vector DB) | Medium–high (data prep, GPUs, eval) |
| Per-request cost | Higher (longer prompts) | Can be lower (shorter prompts, smaller model) |
| Hallucination control | Grounding + "I don't know" | Can still hallucinate confidently |
| Access control | Filter docs per user | Can't un-teach per user |
| Risks | Retrieval misses, prompt injection via docs | Catastrophic forgetting, overfitting |
| Availability | Works with any API model | Needs open weights or a provider tuning service |

### Decision guide
1. **Start with prompting** (clear system prompt, few-shot examples, structured output).
2. Model lacks **knowledge** → add **RAG**.
3. Model has knowledge but **behaves inconsistently** (format, tone, specialized labels) or you need a **smaller/cheaper model** to match a big one on one task → **fine-tune** (LoRA/QLoRA for efficiency).
4. Need both → **fine-tuned model + RAG** (e.g., a model tuned to always cite in your format, fed by retrieval).

### Can RAG replace fine-tuning?
For knowledge questions, usually yes. For behavior — strict output schema, domain-specific classification, compressing a big model's skill into a small one — no; that is what fine-tuning is for.

## Example
**Analogy:** A new employee. **RAG** = giving them access to the company wiki; they look things up and quote it. **Fine-tuning** = sending them to a training course; they internalize *how* to work (writing style, procedures) but won't know next month's new policy unless retrained.

**Python – a simple router that encodes the decision guide:**
```python
def choose_approach(needs_fresh_or_private_facts: bool,
                    needs_citations: bool,
                    needs_consistent_style_or_format: bool,
                    prompting_already_tried: bool) -> str:
    knowledge = needs_fresh_or_private_facts or needs_citations
    behavior = needs_consistent_style_or_format and prompting_already_tried
    if knowledge and behavior:
        return "RAG + fine-tuning"
    if knowledge:
        return "RAG"
    if behavior:
        return "Fine-tuning (try LoRA/QLoRA)"
    return "Prompt engineering first"

print(choose_approach(True, True, False, False))   # RAG
print(choose_approach(False, False, True, True))   # Fine-tuning (try LoRA/QLoRA)
print(choose_approach(True, False, True, True))    # RAG + fine-tuning
```

## Common Mistakes / Misconceptions
- **Fine-tuning to add facts** that change weekly — you'll be retraining forever and can't cite sources.
- **Using RAG to fix formatting/tone problems** that a few examples or a fine-tune would solve.
- **Skipping prompt engineering** before investing in either.
- **Thinking they are mutually exclusive** — they compose well.

## Key Takeaways
- RAG injects knowledge at runtime; fine-tuning changes weights to shape behavior.
- RAG wins on freshness, citations, and per-user access control.
- Fine-tuning wins on consistent style/format, specialized tasks, and shrinking prompts/models.
- Escalate: prompting → RAG → fine-tuning → both.

## Related Questions
- [How is RAG different from fine-tuning?](../Week-08-Retrieval-Augmented-Generation-RAG/03-How-is-RAG-different-from-fine-tuning.md)
- [Can RAG replace fine-tuning?](../Week-09-Fine-tuning-and-Model-Optimization/15-Can-RAG-replace-fine-tuning.md)
- [When should you fine-tune a model?](../Week-09-Fine-tuning-and-Model-Optimization/03-When-should-you-fine-tune-a-model.md)
- [What is catastrophic forgetting?](../Week-09-Fine-tuning-and-Model-Optimization/13-What-is-catastrophic-forgetting.md)
- [Why do LLMs need external knowledge?](../Week-08-Retrieval-Augmented-Generation-RAG/02-Why-do-LLMs-need-external-knowledge.md)
- [What is prompt engineering?](../Week-06-Working-with-LLMs-and-Model-APIs/09-What-is-prompt-engineering.md)
- [Explain the complete RAG pipeline.](05-Explain-the-complete-RAG-pipeline.md)
- [Explain LoRA and QLoRA.](07-Explain-LoRA-and-QLoRA.md)
