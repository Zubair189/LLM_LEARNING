# What is few-shot prompting?

> Bonus question (added) · Week 06 – Working with LLMs and Model APIs

## In One Line
Few-shot prompting means **putting a few worked examples in your prompt** so the model copies the pattern.

## Simple Explanation
Sometimes it's hard to explain in words exactly what you want. It's often easier to **show** it.

In few-shot prompting you add a few examples of **input → correct output** before your real question. The model spots the pattern and follows it.

The names:
1. **Zero-shot:** only instructions, no examples.
2. **One-shot:** one example.
3. **Few-shot:** a few examples, usually 2 to 5.

Example:
- "Great product!" → positive
- "Broke after a day." → negative
- "It's okay." → neutral
- "Slow delivery." → ?

The model sees the pattern and answers with one of the same labels, in the same format.

Tips:
- Use **different kinds** of examples, including tricky ones.
- **Balance** the labels, so the model doesn't lean towards one answer.
- Keep the **format identical** in every example.
- Still give a short, clear instruction.

You can put examples **inside one prompt**, or as fake earlier **user / assistant** turns in the `messages` list.

**Link to LLMs:** this is called **in-context learning**. The model doesn't change; it just uses the examples in this one request.

## Real-Life Example
Before a new barista starts, you show them three finished order cards. They pick up the labelling style much faster than by reading a long manual.

## Code Example
Building a few-shot prompt (you would then send it as the user message).

```python
examples = [                                        # a few worked examples (the "shots")
    ("Great product, arrived early!", "positive"),
    ("Broke after one day.", "negative"),
    ("It's fine, nothing special.", "neutral"),
]

def few_shot_prompt(query):
    lines = ["Classify the review as positive, negative or neutral."]  # instruction
    for text, label in examples:                    # add every example
        lines.append(f"Review: {text}\nLabel: {label}")
    lines.append(f"Review: {query}\nLabel:")        # the new one, answer left blank
    return "\n\n".join(lines)

print(few_shot_prompt("Love it!"))

# Output:
# Classify the review as positive, negative or neutral.
#
# Review: Great product, arrived early!
# Label: positive
#
# Review: Broke after one day.
# Label: negative
#
# Review: It's fine, nothing special.
# Label: neutral
#
# Review: Love it!
# Label:
```

**What happened:** we built a prompt with three examples and left the last label blank. A model receiving it would very likely reply "positive".

## Watch Out
- Few-shot does **not** train the model. The examples only count for that one request.
- Every example costs tokens on every call. Use a few good ones, not dozens.

## Remember
- Few-shot = **show a few examples** in the prompt.
- Great for **consistent labels and formats**.
- Keep examples **varied, balanced and in the same format**.

## Related Questions
- [What is prompt engineering?](09-What-is-prompt-engineering.md)
- [What is a system prompt?](10-What-is-a-system-prompt.md)
- [What is a user prompt?](11-What-is-a-user-prompt.md)
- [What is structured output?](15-What-is-structured-output.md)
