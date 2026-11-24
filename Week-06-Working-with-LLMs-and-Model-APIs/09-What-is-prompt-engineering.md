# What is prompt engineering?

> Week 06 – Working with LLMs and Model APIs · Question 9

## Short Answer
Prompt engineering is the practice of designing and iteratively testing the instructions, context and examples you give an LLM so it reliably produces the output you want. It covers clear task descriptions, roles (system prompts), examples (few-shot), output format specs, and step-by-step reasoning requests. It is cheaper and faster than fine-tuning and is usually the first lever an LLM engineer pulls.

## Detailed Explanation
A well-engineered prompt usually contains:

| Element | Example |
|---|---|
| Role / context | "You are a support agent for an e-commerce store." |
| Task | "Classify the customer message into one category." |
| Input data, clearly delimited | `<message>...</message>` (XML tags or headings) |
| Constraints | "Only use categories: billing, shipping, other." |
| Output format | "Reply with JSON: {\"category\": ...}" |
| Examples (few-shot) | 2–5 input → output pairs |
| Reasoning guidance | "Think through the policy before answering." |

Core techniques:
- **Be specific and explicit** – vague prompts get vague answers. Say what *to* do, not only what not to do.
- **Separate instructions from data** – delimiters reduce confusion and prompt-injection risk.
- **Few-shot examples** – show the pattern you want ([bonus question](18-BONUS-What-is-few-shot-prompting.md)).
- **Chain of thought / extended thinking** – ask the model to reason before answering for multi-step problems.
- **Decompose** – split a big task into a chain of smaller prompts.
- **Iterate with an eval set** – keep 10–50 test inputs with expected outputs and measure each prompt change.

```
 write prompt ─► run on test set ─► inspect failures ─► revise ─┐
       ▲                                                        │
       └────────────────────────────────────────────────────────┘
```

## Example
**Analogy:** Prompting is like briefing a smart new contractor: the clearer the brief (goal, context, examples, deliverable format), the better the result.

```python
import anthropic

client = anthropic.Anthropic()

vague = "What category is this: 'I was charged twice for my order'"
engineered = """Classify the customer message into exactly one category.
Categories: billing, shipping, account, other.

<message>
I was charged twice for my order
</message>

Reply with only the category name in lowercase."""

for p in (vague, engineered):
    r = client.messages.create(model="claude-haiku-5-5", max_tokens=100,
                               messages=[{"role": "user", "content": p}])
    print(repr(r.content[0].text))
# Expected (vague): a sentence like 'This falls under billing / payment issues ...'
# Expected (engineered): 'billing'  -> easy to parse in code
```

Simple prompt evaluation:
```python
tests = [("Where is my parcel?", "shipping"), ("Reset my password", "account")]
def classify(msg):
    r = client.messages.create(model="claude-haiku-5-5", max_tokens=10,
        messages=[{"role": "user", "content": engineered.replace(
            "I was charged twice for my order", msg)}])
    return r.content[0].text.strip()
print(sum(classify(m) == y for m, y in tests), "/", len(tests))  # Expected: 2 / 2
```

## Common Mistakes / Misconceptions
- **"Prompt engineering is magic phrases."** It is mostly clear communication plus systematic testing.
- **Changing prompts without an eval set** – you fix one case and break three others.
- **Mixing untrusted user text with instructions** without delimiters, inviting prompt injection.
- **Overlong prompts full of contradictions** – more words are not always better.

## Key Takeaways
- Prompt engineering = designing + testing inputs to get reliable outputs.
- Include role, task, delimited data, constraints, format and examples.
- Iterate against a small evaluation set.
- Try prompting before fine-tuning.

## Related Questions
- [What is a system prompt?](10-What-is-a-system-prompt.md)
- [What is a user prompt?](11-What-is-a-user-prompt.md)
- [What is few-shot prompting?](18-BONUS-What-is-few-shot-prompting.md)
- [What is structured output?](15-What-is-structured-output.md)
