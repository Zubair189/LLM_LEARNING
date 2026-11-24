# What is hallucination detection?

> Week 11 – Production LLM Applications · Question 14

## Short Answer
Hallucination detection means automatically spotting when an LLM states things that are false or unsupported by its sources. Common techniques: check each claim against the retrieved context (groundedness/faithfulness), require and verify citations, use a second model as a judge, sample several answers and flag disagreement (self-consistency), and validate facts against databases or tools. Detected hallucinations can be blocked, regenerated, flagged with a warning or sent for human review.

## Detailed Explanation
**Two kinds of hallucination**
- **Unfaithful (to context):** in RAG, the answer says something the retrieved documents don't support.
- **Factually wrong (to the world):** fabricated facts, citations, APIs or numbers.

**Detection techniques**
| Technique | How it works | Cost |
|---|---|---|
| Citation checks | Require `[doc_id]` citations; verify the cited text actually supports the sentence | Low |
| Groundedness judge | An LLM compares answer vs. context and labels each claim SUPPORTED/UNSUPPORTED | Medium |
| NLI models | Entailment model checks whether context entails each claim | Low–medium |
| Self-consistency | Generate N answers; low agreement means likely hallucination | High (N calls) |
| Tool/DB verification | Check numbers, names and dates against a source of truth | Varies |
| Structured output validation | Check that IDs, URLs and enums exist | Low |

```
question ─► retrieve ─► generate answer + citations
                               │
                     ┌─────────▼──────────┐
                     │ grounding checker   │ (judge / NLI / rules)
                     └───┬──────────┬──────┘
                 grounded│          │unsupported claims
                         ▼          ▼
                     return     regenerate / "I'm not sure" / human review
```

Prevention complements detection: ground answers in retrieved context, tell the model to say "I don't know", and lower temperature for factual tasks.

## Example
**Analogy:** A newspaper fact-checker reads a reporter's draft and highlights every claim that isn't backed by a source before it goes to print.

```python
import json
import anthropic

client = anthropic.Anthropic()

def check_grounding(context: str, answer: str) -> dict:
    r = client.messages.create(
        model="claude-haiku-5-5", max_tokens=300, temperature=0,
        system=("You are a strict fact-checker. Using ONLY the context, list each claim in the "
                'answer and label it. Reply with JSON only: {"claims":[{"text":..,"supported":true|false}]}'),
        messages=[{"role": "user",
                   "content": f"<context>{context}</context>\n<answer>{answer}</answer>"}])
    return json.loads(r.content[0].text)

ctx = "The Eiffel Tower is 330 m tall and was completed in 1889."
ans = "The Eiffel Tower is 330 m tall, was completed in 1889, and has 5 restaurants."
result = check_grounding(ctx, ans)
unsupported = [c["text"] for c in result["claims"] if not c["supported"]]
print(unsupported)
# -> ['The Eiffel Tower has 5 restaurants']   (illustrative)
```

## Common Mistakes / Misconceptions
- Thinking RAG eliminates hallucinations. The model can still ignore or misread the context.
- Using the *same* prompt and model to both answer and judge. Use a separate, strict judge prompt.
- Treating detection as perfect. Judges have false positives and negatives, so calibrate on labelled examples.
- Checking only "is it fluent?". Fluent answers are often the most convincing hallucinations.

## Key Takeaways
- Hallucination = unsupported or false content; measure faithfulness to sources.
- Combine citations, grounding judges, NLI, consistency and tool verification.
- Decide on an action: block, regenerate, warn or escalate.
- Prevention (grounding, "I don't know") and detection work together.

## Related Questions
- [Why do LLMs sometimes hallucinate?](../Week-02-Introduction-to-LLMs-and-Generative-AI/12-Why-do-LLMs-sometimes-hallucinate.md)
- [What is hallucination in RAG?](../Week-08-Retrieval-Augmented-Generation-RAG/12-What-is-hallucination-in-RAG.md)
- [What is LLM evaluation?](13-What-is-LLM-evaluation.md)
- [What are guardrails in LLM applications? (bonus)](19-BONUS-What-are-guardrails-in-LLM-applications.md)
