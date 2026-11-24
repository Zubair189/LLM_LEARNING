# What is model pretraining?

> Week 09 – Fine-tuning and Model Optimization · Question 1

## In One Line
Pretraining is the first, biggest stage of building an LLM, where the model reads a huge amount of text and learns by **guessing the next word** again and again.

## Simple Explanation
An LLM starts as a big network of random numbers (called **weights**). It knows nothing yet.

During pretraining, we show it trillions of words from websites, books and code. For each piece of text, the model tries to guess the next word (really the next **token**, a small piece of a word). If the guess is wrong, its weights are nudged a little so the next guess is better.

1. Take a sentence: "The capital of France is ___".
2. The model guesses a word.
3. The real text says "Paris", so we check the guess and adjust the weights.
4. Repeat this billions of times.

Nobody has to write the answers. The text itself contains them, because the next word is always there. This is called **self-supervised learning**.

The result is a **base model**. It knows a lot about language, facts and code, but it is not good at following instructions yet. Ask it a question and it may just continue writing more questions. Later steps (fine-tuning) teach it to behave like a helpful assistant.

Pretraining is extremely expensive: thousands of GPUs for weeks, costing millions of dollars. That's why almost everyone starts from a model someone else already pretrained.

## Real-Life Example
Pretraining is like a student who spends years reading a whole library. They know a huge amount, but nobody has taught them how to answer an exam question neatly. Fine-tuning is the short exam-prep course afterwards.

## Code Example
A tiny "next-word guesser" that learns only by reading text, with no labels.

```python
text = "the cat sat on the mat . the cat ate . the dog sat on the rug ."
words = text.split()                       # split the text into words

counts = {}                                # word -> {next word: how often}
for current, nxt in zip(words, words[1:]): # look at every pair of neighbours
    counts.setdefault(current, {})
    counts[current][nxt] = counts[current].get(nxt, 0) + 1

def guess_next(word):
    options = counts[word]                 # all words seen after this word
    return max(options, key=options.get)   # pick the most common one

print(guess_next("the"))   # cat
print(guess_next("sat"))   # on
```

**What happened:** the program learned "after *the* usually comes *cat*" just by reading. A real LLM does the same kind of next-word learning, but with billions of weights and trillions of words.

## Watch Out
- Pretraining does **not** use hand-labelled data. The next word *is* the answer.
- A base model is **not a chatbot** yet. It needs fine-tuning to follow instructions.

## Remember
- Pretraining = learn by guessing the next word on huge amounts of text.
- It is the most expensive stage and gives the model its general knowledge.
- Its output is a base model; instruction-following comes later.

## Related Questions
- [What is fine-tuning?](02-What-is-fine-tuning.md)
- [What is supervised fine-tuning (SFT)?](04-What-is-supervised-fine-tuning-SFT.md)
- [What is preference tuning (RLHF and DPO)?](16-BONUS-What-is-preference-tuning-RLHF-and-DPO.md)
- [What is a training dataset? (Week 01)](../Week-01-Introduction-to-Artificial-Intelligence/10-What-is-a-training-dataset.md)
