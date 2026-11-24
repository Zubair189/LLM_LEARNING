# What are training examples and labels?

> Week 09 – Fine-tuning and Model Optimization · Question 6

## In One Line
A **training example** is something the model is shown, and its **label** is the correct answer the model should learn to give for it.

## Simple Explanation
In classic machine learning this is easy to see:
- Example: an email. Label: "spam" or "not spam".
- Example: a photo. Label: "cat".

For LLMs, it works the same way, just with words:

1. **Pretraining:** the example is a piece of text, and the label is simply **the next word** at each position. "The cat sat" → the label after "The" is "cat", after "cat" is "sat".
2. **Fine-tuning (SFT):** the example is a whole chat. The labels are the words of the **assistant's reply**. The user's part is marked with `-100`, a special value that means "don't grade this".
3. **Preference tuning (DPO):** the example is a prompt, and instead of one label there are two replies: a **better** one and a **worse** one.

Good labels are:
- **Correct** and in the exact format you want.
- **Consistent** across examples.
- **Complete**, including an "end" token so the model learns when to stop talking.

## Real-Life Example
Flashcards: the front of the card is the example ("Capital of Japan?"), the back is the label ("Tokyo"). You learn by looking at the front, guessing, and checking the back.

## Code Example
See how one sentence becomes many "guess the next word" examples.

```python
words = ["The", "cat", "sat", "<end>"]   # one short training text

# Each position's label is simply the word that comes next
for i in range(len(words) - 1):
    example = words[: i + 1]              # what the model has seen so far
    label = words[i + 1]                  # the word it should predict
    print(example, "->", label)

# Output:
# ['The'] -> cat
# ['The', 'cat'] -> sat
# ['The', 'cat', 'sat'] -> <end>
```

**What happened:** a single 4-word text gave us 3 small examples with labels for free. That is why LLMs can learn from raw text without humans writing labels.

## Watch Out
- In LLMs, the label is not one word for the whole example. It is **the next token at every position**.
- Wrong or inconsistent labels teach the model wrong habits. It copies what you show it.

## Remember
- Example = input; label = correct output.
- In pretraining, the label is the next word; in SFT, it is the assistant's reply.
- Clean, consistent labels make a better model.

## Related Questions
- [What is a training dataset?](05-What-is-a-training-dataset.md)
- [What is supervised fine-tuning (SFT)?](04-What-is-supervised-fine-tuning-SFT.md)
- [What is preference tuning (RLHF and DPO)?](16-BONUS-What-is-preference-tuning-RLHF-and-DPO.md)
- [What is model pretraining?](01-What-is-model-pretraining.md)
