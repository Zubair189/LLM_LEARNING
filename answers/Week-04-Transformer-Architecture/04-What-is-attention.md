# What is attention?

> Week 04 – Transformer Architecture · Question 4

## In One Line
Attention lets a model **focus on the most relevant pieces of information** by giving each piece a weight and blending them.

## Simple Explanation
When you read "The cat sat on the mat because **it** was tired", you know "it" means the cat. You "paid attention" to the right word. Attention gives a model the same ability.

It uses three things:
- **Query:** what I'm looking for.
- **Key:** a label on each piece of information, used for matching.
- **Value:** the actual content of each piece.

**The three steps:**
1. **Score:** compare the query with every key using a **dot product** (multiply matching numbers and add). A high score means "very relevant".
2. **Turn scores into weights:** apply **softmax**, which makes them positive and add up to 1 (like percentages).
3. **Blend:** take a weighted mix of the values. More relevant pieces contribute more.

The result is a "soft lookup": instead of picking one item, it takes a bit of everything, mostly from the best matches.

In real Transformers, the scores are also divided by √d (the square root of the vector length) before softmax. This keeps the numbers from getting too big.

**Link to LLMs:** attention is the core operation inside every Transformer layer. When the query, keys and values all come from the same sentence, it is called **self-attention** (next question).

## Real-Life Example
In a library, your question is the **query**, each book's title is a **key**, and the book's content is the **value**. Instead of borrowing one book, you skim all of them, spending more time on the books whose titles best match your question, and combine what you learn.

## Code Example
One query, three items (scaling skipped to keep the numbers simple):

```python
import math

q = [1, 0]                                   # query: what I'm looking for
keys   = [[1, 0], [0, 1], [0.7, 0.7]]        # one key per item
values = [[10, 0], [0, 10], [5, 5]]          # one value per item

scores = [sum(a * b for a, b in zip(q, k)) for k in keys]   # step 1: dot products
exps = [math.exp(s) for s in scores]
weights = [e / sum(exps) for e in exps]                     # step 2: softmax
output = [sum(w * v[i] for w, v in zip(weights, values)) for i in range(2)]  # step 3

print(scores)                               # [1, 0, 0.7]
print([round(w, 2) for w in weights])       # [0.47, 0.17, 0.35]
print([round(o, 2) for o in output])        # [6.5, 3.5]
```

**What happened:** the first key matched the query best, so it got the biggest weight (0.47), and the output leans towards its value `[10, 0]`.

## Watch Out
- Attention does **not** pick just one item; it makes a weighted blend.
- Attention weights are a helpful clue, but not a full explanation of why a model made a decision.

## Remember
- Attention = **score, softmax, blend**.
- Query = what I want, Key = label, Value = content.
- It is the heart of the Transformer.

## Related Questions
- [What is self-attention?](../Week-04-Transformer-Architecture/05-What-is-self-attention.md)
- [What are Query, Key and Value?](../Week-04-Transformer-Architecture/06-What-are-Query-Key-and-Value.md)
- [What is an attention score?](../Week-04-Transformer-Architecture/07-What-is-an-attention-score.md)
- [What is softmax?](../Week-03-Mathematics-and-Neural-Network-Foundations/10-What-is-softmax.md)
