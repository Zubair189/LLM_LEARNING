# What are Query, Key and Value?

> Week 04 – Transformer Architecture · Question 6

## In One Line
For every word, the model makes three vectors: a **Query** (what am I looking for?), a **Key** (how can I be found?) and a **Value** (what information do I pass on?).

## Simple Explanation
Attention needs each word to play three different roles. So the model takes each word's embedding (its list of numbers) and multiplies it by three different **learned weight matrices**, called `W_Q`, `W_K` and `W_V`. This gives three new vectors:

- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain? How can others find me?"
- **Value (V):** "If you pick me, what information do I give you?"

**How they are used:**
1. Compare one word's **query** with every word's **key** (dot product). This gives relevance scores.
2. Softmax turns the scores into weights that add up to 1.
3. Mix the **values** using those weights.

**Why three separate vectors instead of one?**
- **Looking and being found are different jobs.** "It" is looking for a noun; "animal" is advertising that it *is* a noun. Different roles need different vectors.
- **Matching is not the same as content.** The features useful for *finding* a word are not always the information worth *copying* from it.
- **They are learned.** Training adjusts `W_Q`, `W_K` and `W_V` so the model discovers which relationships matter.

**Link to LLMs:** when an LLM generates text, the keys and values of earlier words never change, so they are saved and reused. This is the **KV cache** (see the bonus question).

## Real-Life Example
Searching on a video website. What you type is the **query**. Each video's title and tags are its **key**. The video itself is the **value**. The site matches your query against the keys and shows you the best-matching content.

## Code Example
```python
def project(x, W):                     # vector x times matrix W
    return [sum(x[i] * W[i][j] for i in range(len(x))) for j in range(len(W[0]))]

animal, it = [1, 0], [0, 1]            # tiny embeddings
W_Q = [[0, 1], [1, 0]]                 # learned weights (made up here)
W_K = [[1, 0], [0, 1]]
W_V = [[2, 0], [0, 2]]

q_it = project(it, W_Q)                # what "it" is looking for
k_animal = project(animal, W_K)        # how "animal" can be found
k_it = project(it, W_K)
v_animal = project(animal, W_V)        # what "animal" will hand over

print(q_it, k_animal, v_animal)                     # [1, 0] [1, 0] [2, 0]
print(sum(a * b for a, b in zip(q_it, k_animal)))   # 1  good match
print(sum(a * b for a, b in zip(q_it, k_it)))       # 0  no match
```

**What happened:** the weights turned "it" into a query that matches the key of "animal". So "it" would pull in the value of "animal".

## Watch Out
- In self-attention, Q, K and V are **not** different inputs. They are three versions of the **same** word, made with different weights.
- Keys decide **how much** to attend; values decide **what** gets passed on.

## Remember
- Q = looking for, K = label for matching, V = content.
- Score = Query · Key; output = weighted mix of Values.
- Keys and values of past words are cached during generation.

## Related Questions
- [What is attention?](../Week-04-Transformer-Architecture/04-What-is-attention.md)
- [What is an attention score?](../Week-04-Transformer-Architecture/07-What-is-an-attention-score.md)
- [What is multi-head attention?](../Week-04-Transformer-Architecture/08-What-is-multi-head-attention.md)
- [What is the KV cache and why does it speed up generation? (bonus)](../Week-04-Transformer-Architecture/16-BONUS-What-is-the-KV-cache-and-why-does-it-speed-up-generation.md)
