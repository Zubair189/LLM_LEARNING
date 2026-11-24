# What is few-shot prompting?

> Bonus question (added) · Week 06 – Working with LLMs and Model APIs

## Short Answer
Few-shot prompting means including a small number of worked examples (input → desired output) in the prompt so the model infers the task, format and style from them — no training required. Zero-shot gives only instructions; one-shot gives one example; few-shot typically gives 2–5. It is one of the most effective ways to get consistent labels, formats or tone, and is a form of in-context learning.

## Detailed Explanation
```
 Zero-shot:  "Classify sentiment: 'Slow delivery.'"
 Few-shot:   "Great product!"      → positive
             "Broke after a day."  → negative
             "It's okay I guess."  → neutral
             "Slow delivery."      → ?
```

Why it works: LLMs are pattern continuers. Examples show *exactly* what "good" looks like — label vocabulary, casing, length, edge-case handling — which is often hard to describe in words.

Ways to supply examples:
- **Inline in one prompt**, clearly delimited (e.g. `<example>` tags).
- **As fake prior turns**: alternating `user`/`assistant` messages showing input and ideal reply.

Best practices:
| Tip | Why |
|---|---|
| Use 2–5 diverse, representative examples | Covers variation without bloating tokens |
| Include edge cases (sarcasm, empty input) | Teaches tricky behaviour |
| Balance labels | Avoids biasing toward the most common example label |
| Keep format identical across examples | Model copies format precisely |
| Combine with clear instructions | Examples show *how*, instructions say *what/why* |

Trade-offs: every example costs input tokens on every call (prompt caching helps), and the model may over-copy surface features of examples. If you need hundreds of examples to get it right, consider fine-tuning or retrieval of examples ("dynamic few-shot").

## Example
**Analogy:** Showing a new barista three finished latte orders on cards before they start — they learn the labelling style faster than from a written manual.

Few-shot as prior conversation turns:
```python
import anthropic

client = anthropic.Anthropic()

examples = [
    ("Great product, arrived early!", "positive"),
    ("Broke after one day.", "negative"),
    ("It's fine, nothing special.", "neutral"),
]
messages = []
for text, label in examples:
    messages.append({"role": "user", "content": text})
    messages.append({"role": "assistant", "content": label})
messages.append({"role": "user", "content": "Delivery was slow but support was lovely."})

resp = client.messages.create(
    model="claude-haiku-5-5",
    max_tokens=5,
    system="Classify review sentiment. Reply with one lowercase word: positive, negative or neutral.",
    messages=messages,
)
print(resp.content[0].text)
# Expected: neutral   (or 'positive' — mixed reviews are the edge case to add as an example)
```

Inline template version (works with any model, including local ones):
```python
def few_shot_prompt(examples, query):
    shots = "\n".join(f"<example>\nInput: {i}\nOutput: {o}\n</example>" for i, o in examples)
    return f"{shots}\nInput: {query}\nOutput:"

print(few_shot_prompt(examples[:1], "Love it"))
# Expected:
# <example>
# Input: Great product, arrived early!
# Output: positive
# </example>
# Input: Love it
# Output:
```

## Common Mistakes / Misconceptions
- **"Few-shot trains the model."** Weights don't change; examples only live in that request's context.
- **All examples share one label or length**, biasing outputs.
- **Examples contradicting the instructions** — the model often follows the examples.
- **Too many examples** — higher cost and latency with diminishing returns.

## Key Takeaways
- Few-shot = a handful of input→output examples in the prompt.
- Great for consistent formats, labels and tone.
- Supply examples inline or as prior user/assistant turns.
- Keep examples diverse, balanced and format-consistent; watch token cost.

## Related Questions
- [What is prompt engineering?](09-What-is-prompt-engineering.md)
- [What is a system prompt?](10-What-is-a-system-prompt.md)
- [What is a user prompt?](11-What-is-a-user-prompt.md)
- [What is structured output?](15-What-is-structured-output.md)
