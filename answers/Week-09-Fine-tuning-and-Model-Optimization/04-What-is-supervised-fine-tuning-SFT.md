# What is supervised fine-tuning (SFT)?

> Week 09 – Fine-tuning and Model Optimization · Question 4

## In One Line
SFT teaches a model by showing it many **question → ideal answer** pairs, so it learns to answer the same way.

## Simple Explanation
"Supervised" means every example comes with the correct answer, written by a person or by a strong model. It's like learning from worked examples.

How it works:

1. Collect pairs: a user message and the exact reply you want.
2. Show each pair to the model.
3. The model tries to produce the reply, word by word (token by token).
4. We measure how far its guess is from the ideal reply (this measure is called the **loss**) and adjust the weights a little.

One important trick: we only grade the model on the **answer part**, not on the question. The question tokens are hidden from the grading (they get a special label `-100`, which means "ignore"). That way the model learns to *answer*, not to repeat the question.

SFT is the step that turns a base model (which only continues text) into an assistant that follows instructions. Order: pretraining → **SFT** → preference tuning.

The data is usually stored as **JSONL**: one example per line, written as a chat:

```json
{"messages":[{"role":"user","content":"My order 123 hasn't arrived"},{"role":"assistant","content":"{\"intent\":\"late_delivery\",\"order_id\":\"123\"}"}]}
```

## Real-Life Example
SFT is like an apprenticeship. The master says "here's a customer request, and here's exactly how I would reply", hundreds of times, until the apprentice replies the same way.

## Code Example
How the "only grade the answer" trick looks.

```python
prompt_tokens = ["<user>", "Hi", "<assistant>"]   # the question part
answer_tokens = ["Hello", "!", "<end>"]           # the part we want to teach

tokens = prompt_tokens + answer_tokens            # what the model reads
labels = [-100] * len(prompt_tokens) + answer_tokens  # -100 = "don't grade this"

for t, l in zip(tokens, labels):
    print(f"{t:12} -> {l}")

# Output:
# <user>       -> -100
# Hi           -> -100
# <assistant>  -> -100
# Hello        -> Hello
# !            -> !
# <end>        -> <end>
```

**What happened:** the question tokens are marked `-100`, so only "Hello ! <end>" count when we measure mistakes. Real libraries (like TRL's `SFTTrainer`) do this for you, using token IDs instead of words.

## Watch Out
- SFT copies whatever you show it, mistakes included. Clean data matters more than lots of data.
- Use the same chat format in training and in real use, or answers can look broken.

## Remember
- SFT = learn from example question → answer pairs.
- Only the answer part is graded.
- About 1,000 excellent examples can beat 50,000 messy ones.

## Related Questions
- [What are training examples and labels?](06-What-are-training-examples-and-labels.md)
- [What is a training dataset?](05-What-is-a-training-dataset.md)
- [What is preference tuning (RLHF and DPO)?](16-BONUS-What-is-preference-tuning-RLHF-and-DPO.md)
- [What is QLoRA?](08-What-is-QLoRA.md)
