# What are positional encodings?

> Week 04 – Transformer Architecture · Question 9

## In One Line
Positional encodings **tell a Transformer the order of the words**, because attention on its own cannot tell first from last.

## Simple Explanation
"Dog bites man" and "man bites dog" have the same words but very different meanings. Attention compares words with each other, but it has **no built-in sense of order**. If you shuffle the words, each word's attention result stays the same, just shuffled. So we must add position information.

There are two main ways:

1. **Add a position code to each word (original Transformer, 2017).** Each position (0, 1, 2, …) gets its own list of numbers made from **sine and cosine waves**. This code is added to the word's embedding. Some pairs of numbers change quickly from one position to the next, others change slowly, a bit like the second, minute and hour hands of a clock. Together they give every position a unique pattern. (BERT and GPT-2 instead *learned* one position vector per position.)

2. **RoPE – Rotary Position Embedding (most modern LLMs).** Instead of adding a code, RoPE **rotates** each word's query and key vectors by an angle that depends on its position. When two rotated vectors are compared, the result depends on **how far apart** the words are, not where they are exactly. That "relative distance" is usually what matters in language.

**Link to LLMs:** Llama, Qwen, Mistral and most current LLMs use RoPE. How positions are handled also affects how long a text (context) a model can work with.

## Real-Life Example
Seats in a theatre. Two people might look identical, but their seat numbers tell you who sits next to whom. RoPE is like giving each person a compass needle turned by their seat number: comparing two needles tells you how many seats apart they are.

## Code Example
The original sine/cosine position codes, with 4 numbers per position:

```python
import math

def position_code(pos, d=4):
    code = []
    for i in range(0, d, 2):                      # pairs of numbers
        speed = 1 / (10000 ** (i / d))            # each pair ticks at a different speed
        code += [round(math.sin(pos * speed), 2), round(math.cos(pos * speed), 2)]
    return code

for pos in range(4):
    print(pos, position_code(pos))
# 0 [0.0, 1.0, 0.0, 1.0]
# 1 [0.84, 0.54, 0.01, 1.0]
# 2 [0.91, -0.42, 0.02, 1.0]
# 3 [0.14, -0.99, 0.03, 1.0]
```

**What happened:** every position got a different pattern. The first pair changes fast (a "seconds hand"), the second pair changes slowly (an "hour hand").

## Watch Out
- Transformers do **not** know word order automatically. Without position information, order is lost.
- RoPE is applied to queries and keys inside attention, not added to the embeddings at the start.

## Remember
- Attention is order-blind; positions must be added.
- Original method: add sine/cosine codes to embeddings.
- Modern method: RoPE rotates Q and K so scores depend on distance.

## Related Questions
- [What are token embeddings?](../Week-04-Transformer-Architecture/10-What-are-token-embeddings.md)
- [What is self-attention?](../Week-04-Transformer-Architecture/05-What-is-self-attention.md)
- [What is a Transformer?](../Week-04-Transformer-Architecture/02-What-is-a-Transformer.md)
