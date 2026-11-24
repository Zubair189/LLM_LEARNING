# What is a training dataset?

> Week 09 – Fine-tuning and Model Optimization · Question 5

> **Note:** Week 01 Q10 explains what a training dataset is in general ([link](../Week-01-Introduction-to-Artificial-Intelligence/10-What-is-a-training-dataset.md)); this answer is about the special kind used to **fine-tune an LLM**.

## In One Line
A fine-tuning dataset is a collection of example conversations that show the model **exactly what input it will get and what answer you want back**.

## Simple Explanation
For fine-tuning, each example is usually a small chat: a user message and the ideal assistant reply. They are saved in a **JSONL** file, which simply means one example per line, written in JSON.

What makes a good dataset:
- **Correct:** every answer is right and in the exact format you want.
- **Consistent:** the same kind of question is always answered the same way.
- **Varied:** it covers the different ways real users ask, including tricky cases.
- **Clean:** no duplicates, no private personal data.

**How much data?** Quality beats quantity. For teaching a format or style with LoRA, a few hundred to about a thousand good examples is often enough. A narrow task may need a few thousand.

**Split the data into three parts:**
1. **Train** (about 80–90%): the model learns from these.
2. **Validation** (about 5–10%): checked during training to spot problems early.
3. **Test** (about 5–10%): kept hidden until the very end to give an honest score.

Remove duplicates *before* splitting, so the same example can't sneak into both train and test.

## Real-Life Example
Think of a cooking school. The training set is the recipes students practise. The validation set is the weekly practice tasting. The test set is the final exam dish, which nobody has seen in advance.

## Code Example
Clean and split a tiny dataset.

```python
import random

examples = [f"example {i}" for i in range(20)] + ["example 3"]  # 21 items, one duplicate
examples = list(dict.fromkeys(examples))   # remove duplicates, keep order -> 20 left

random.seed(0)                             # fixed seed so the shuffle is repeatable
random.shuffle(examples)                   # mix them up before splitting

n = len(examples)
train = examples[: int(n * 0.8)]           # first 80%
val   = examples[int(n * 0.8): int(n * 0.9)]  # next 10%
test  = examples[int(n * 0.9):]            # last 10%

print(len(train), len(val), len(test))     # 16 2 2
```

**What happened:** we removed the duplicate, shuffled, and split 80/10/10. The test examples stay untouched until we evaluate the final model.

## Watch Out
- Never let test examples leak into training, or your score will look better than it really is.
- If you generate examples with another AI, still read a sample by hand.

## Remember
- A fine-tuning dataset = example chats with the ideal reply.
- Quality and consistency matter more than size.
- Always split into train, validation and test.

## Related Questions
- [What is a training dataset? (Week 01 – general definition)](../Week-01-Introduction-to-Artificial-Intelligence/10-What-is-a-training-dataset.md)
- [What are training examples and labels?](06-What-are-training-examples-and-labels.md)
- [What is supervised fine-tuning (SFT)?](04-What-is-supervised-fine-tuning-SFT.md)
- [How do you evaluate a fine-tuned model?](14-How-do-you-evaluate-a-fine-tuned-model.md)
