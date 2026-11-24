# How does tokenization work?

> Week 02 – Introduction to LLMs and Generative AI · Question 7

## In One Line
Tokenization cuts text into pieces using a **vocabulary learned in advance**, where the most common letter combinations get merged into single tokens.

## Simple Explanation
Tokenization is turning text into token numbers the model can use, and turning numbers back into text at the end.

Most LLMs use a method called **BPE (Byte-Pair Encoding)**. It has two parts.

**Part 1: Learning the vocabulary (done once, before the model is trained)**

1. Start with single characters: `l`, `o`, `w`, ...
2. Find the pair of neighbours that appears **most often**, for example `l` + `o`.
3. **Merge** them into a new piece: `lo`.
4. Repeat thousands of times. Common words end up as one piece.

The result is a list of merges and a **vocabulary** (all known pieces, each with an ID number).

**Part 2: Using it on new text**

1. Split the new text into characters.
2. Apply the learned merges in the same order.
3. Look up the ID number of each piece.

Frequent words become **one** token. Rare words become **several** pieces. Modern tokenizers can fall back to raw bytes, so **any** text works, even emoji.

A model and its tokenizer are a **matched pair**: you can't swap the tokenizer without retraining the model.

## Real-Life Example
Think of taking notes. If you write "the" all the time, you invent a shortcut for it. Later "th" + "ing" gets its own shortcut too. BPE does the same: the most common combinations earn their own symbol.

## Code Example
A mini BPE (`_` marks the end of a word).

```python
from collections import Counter

def merge(symbols, pair):                        # join every place where `pair` appears
    out, i = [], 0
    while i < len(symbols):
        if tuple(symbols[i:i + 2]) == pair:
            out.append(symbols[i] + symbols[i + 1]); i += 2
        else:
            out.append(symbols[i]); i += 1
    return out

corpus = "low lower lowest newer newest wider widest".split() * 5
words = [list(w) + ["_"] for w in corpus]        # start: single characters
merges = []
for _ in range(10):                              # learn 10 merges
    pairs = Counter(p for w in words for p in zip(w, w[1:]))   # count neighbour pairs
    best = max(pairs, key=pairs.get)             # most common pair
    merges.append(best)
    words = [merge(w, best) for w in words]

def encode(word):                                # apply merges in learned order
    symbols = list(word) + ["_"]
    for pair in merges:
        symbols = merge(symbols, pair)
    return symbols

print(merges[:3])          # [('w', 'e'), ('l', 'o'), ('r', '_')]
print(encode("lowest"))    # ['lowe', 'st_']                  <- common: few pieces
print(encode("slowly"))    # ['s', 'lo', 'w', 'l', 'y', '_']  <- new word: many pieces
```

**What happened:** the tokenizer learned useful pieces like `lo` and `st_`. A known word became 2 tokens, while a word it had never seen was built from smaller pieces. Real projects use a library such as `tiktoken` or Hugging Face `tokenizers`.

## Watch Out
- **Tokenizers don't just split on spaces.** They split into learned pieces.
- **The tokenizer is not learned during model training.** It is built separately, beforehand, and then fixed.

## Remember
- BPE = repeatedly merge the most common pair.
- Common words → 1 token; rare words → several.
- Each model has its own tokenizer.

## Related Questions
- [What is a token?](./06-What-is-a-token.md)
- [Why do LLMs use tokens instead of complete words?](./08-Why-do-LLMs-use-tokens-instead-of-complete-words.md)
- [What are token embeddings?](../Week-04-Transformer-Architecture/10-What-are-token-embeddings.md)
- [What is a context window?](./09-What-is-a-context-window.md)
