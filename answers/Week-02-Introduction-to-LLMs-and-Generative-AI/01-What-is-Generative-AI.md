# What is Generative AI?

> Week 02 – Introduction to LLMs and Generative AI · Question 1

## In One Line
Generative AI is AI that **creates new things**, like text, images, music or code, instead of only sorting or labelling things.

## Simple Explanation
Most older AI **judges** things. It answers questions like "Is this email spam?" or "Is this a cat or a dog?". This kind is called **discriminative AI**.

**Generative AI** **makes** things. It answers requests like "Write a reply to this email" or "Draw a cat wearing a hat".

How it works, in simple steps:

1. It studies a huge number of examples (text, pictures, sounds).
2. It learns the **patterns**: what usually comes next, what usually goes together.
3. When you ask for something, it builds **new** content that follows those patterns, piece by piece, with a little randomness.

Main kinds you will hear about:

- **LLMs** (like Claude): create text and code, one small piece (a **token**) at a time.
- **Diffusion models:** create images or video by starting from random noise and cleaning it up step by step.

It became popular in the 2020s thanks to better model designs (the **Transformer**), huge amounts of training data, and easy access through APIs.

**Link to LLMs:** every chat assistant you use is generative AI. Its answers sound **plausible**, but that does not guarantee they are **true**.

## Real-Life Example
An art critic can look at a painting and say "that's a Van Gogh" (discriminative). An art student who has studied thousands of paintings can paint a **new** picture in Van Gogh's style (generative).

## Code Example
A tiny text generator. It learns which word follows which, then makes new sentences.

```python
import random
from collections import defaultdict
random.seed(1)                         # fixed seed so results repeat

text = "the cat sat on the mat . the dog sat on the rug . the cat ate the rug .".split()

next_words = defaultdict(list)         # word -> list of words seen after it
for word, nxt in zip(text, text[1:]):
    next_words[word].append(nxt)

word = "the"                           # starting word
output = [word]
for _ in range(7):                     # add 7 more words
    word = random.choice(next_words[word])   # pick a word that followed it
    output.append(word)

print(" ".join(output))
# Output:
# the mat . the cat ate the rug
```

**What happened:** it learned word patterns from the text and produced a sentence that never appeared there as a whole. An LLM does the same basic thing, but with a giant neural network and billions of examples.

## Watch Out
- **Generative AI is not only chatbots.** It also makes images, audio, video and code.
- **It does not look up answers in a database.** It generates them from patterns, so it can sound right and still be wrong.

## Remember
- Generative AI **creates**; discriminative AI **labels**.
- It learns patterns, then makes new content that follows them.
- Output is plausible, not guaranteed correct.

## Related Questions
- [What is a Large Language Model (LLM)?](./02-What-is-a-Large-Language-Model-LLM.md)
- [What is a language model?](./05-What-is-a-language-model.md)
- [Why do LLMs sometimes hallucinate?](./12-Why-do-LLMs-sometimes-hallucinate.md)
- [What is Artificial Intelligence (AI)?](../Week-01-Introduction-to-Artificial-Intelligence/01-What-is-Artificial-Intelligence-AI.md)
