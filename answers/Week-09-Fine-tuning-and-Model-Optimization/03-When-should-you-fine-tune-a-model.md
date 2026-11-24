# When should you fine-tune a model?

> Week 09 – Fine-tuning and Model Optimization · Question 3

## In One Line
Fine-tune only when good prompts and RAG are not enough and you need the model to **behave** in a consistent way, and you have enough good examples to teach it.

## Simple Explanation
Fine-tuning costs time and money, so try cheaper options first. A good order is:

1. **Better prompt:** clear instructions and a few examples.
2. **RAG** (Retrieval-Augmented Generation): let the model look up the right documents when a question comes in. Best for facts.
3. **Fine-tuning** (start with LoRA): when you need to change *how* the model answers.

**Good reasons to fine-tune:**
- Every answer must follow a strict format or brand voice, and prompts keep drifting.
- A narrow task done many times, like sorting support emails. A small fine-tuned model can do it cheaply and quickly.
- Your prompts have become very long. Fine-tuning "bakes in" the instructions, so prompts get shorter.

**Bad reasons to fine-tune:**
- "The model doesn't know our product list." Use RAG instead; it stays up to date.
- "We have 30 examples." Too few. Put them in the prompt as examples.
- "It makes things up." Fine-tuning can even make this worse.

Before you start, also build a small **test set** (examples the model never trains on), so you can prove the fine-tuned model really is better.

**Link to LLMs:** many teams combine both: fine-tune for format and style, and use RAG for facts.

## Real-Life Example
You don't send a new employee on a three-month course to learn today's price list; you just hand them the price sheet (RAG). But you *do* train them for weeks to write reports in the company format (fine-tuning).

## Code Example
A simple rule-of-thumb helper.

```python
def should_finetune(n_examples, needs_fresh_facts, prompt_is_good_enough):
    if needs_fresh_facts:          # facts that change -> look them up instead
        return "Use RAG first"
    if prompt_is_good_enough:      # nothing to fix
        return "Keep prompting"
    if n_examples < 200:           # not enough data to learn from
        return "Collect more data / use few-shot examples"
    return "Fine-tune (start with LoRA)"

print(should_finetune(2000, False, False))  # Fine-tune (start with LoRA)
print(should_finetune(5000, True, False))   # Use RAG first
print(should_finetune(50, False, False))    # Collect more data / use few-shot examples
```

**What happened:** the helper checks the cheaper options first and only says "fine-tune" when there is a real behaviour problem and enough data.

## Watch Out
- Fine-tuning and RAG are not either/or. They work well together.
- Every time the base model is upgraded, you may need to fine-tune and test again.

## Remember
- Order: prompt → RAG → fine-tune.
- Fine-tune for **behaviour, format, cost and speed**, not for facts.
- You need enough good examples plus a separate test set.

## Related Questions
- [What is fine-tuning?](02-What-is-fine-tuning.md)
- [Can RAG replace fine-tuning?](15-Can-RAG-replace-fine-tuning.md)
- [How do you evaluate a fine-tuned model?](14-How-do-you-evaluate-a-fine-tuned-model.md)
- [What is a training dataset?](05-What-is-a-training-dataset.md)
