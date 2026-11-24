# How does GPT use the Transformer architecture?

> Week 04 – Transformer Architecture · Question 14

## In One Line
GPT uses **only the decoder part** of the Transformer: it reads all the text so far and **predicts the next token, one at a time**.

## Simple Explanation
GPT stands for **Generative Pre-trained Transformer**. It keeps the decoder and drops the encoder.

**Inside GPT, step by step:**
1. **Tokens → embeddings:** each token becomes a list of numbers, plus position information.
2. **Many decoder blocks:** each has **masked self-attention** (each token looks only at earlier tokens) and a **feed-forward network**, with residual connections and normalisation to keep training stable.
3. **Output:** the last layer gives a probability for every token in the vocabulary: P(next token | all previous tokens).

**How GPT is trained (three stages):**
1. **Pre-training:** read a huge amount of text (web pages, books, code) and practise predicting the next token.
2. **Fine-tuning on instructions:** learn from examples of "instruction → good answer".
3. **Preference training** (e.g. RLHF): learn which answers people prefer, so it becomes a helpful assistant.

**How GPT writes a reply:**
1. Read the whole prompt in one pass (called **prefill**).
2. Pick the next token from the probabilities (settings like temperature and top-p affect this).
3. Add the token and repeat until a stop token. A **KV cache** saves earlier work.

**Link to LLMs:** Claude, Llama, Gemini, Qwen and Mistral follow this same decoder-only recipe, with small upgrades (such as RoPE positions).

## Real-Life Example
A super well-read autocomplete. Having read a huge part of the internet, it knows which word is likely to come next in almost any situation. Extra training then teaches it to use that skill to be a helpful assistant, not just to continue text.

## Code Example
A pretend "model" that gives next-token probabilities for the whole text so far, and a greedy generation loop:

```python
# Pretend model: next-token probabilities given ALL the text so far
model = {
    "The cat":            {"sat": 0.7, "ran": 0.3},
    "The cat sat":        {"on": 0.9, "down": 0.1},
    "The cat sat on":     {"the": 0.8, "a": 0.2},
    "The cat sat on the": {"mat": 0.6, "sofa": 0.4},
}

text = "The cat"                                # the prompt
while text in model:
    probs = model[text]                         # one "forward pass"
    next_token = max(probs, key=probs.get)      # greedy: take the most likely
    text = text + " " + next_token              # add it and repeat
    print(text)
# The cat sat
# The cat sat on
# The cat sat on the
# The cat sat on the mat
```

**What happened:** each loop is like one forward pass that predicts one token, which is then added to the text. A real GPT computes these probabilities with its Transformer layers instead of a lookup table.

## Watch Out
- GPT does not look answers up in a database; it generates from patterns stored in its weights (unless connected to tools or search).
- Pre-training alone does not make a chatbot; instruction and preference training do.

Note: Week 2 Q4 explains what the name GPT stands for; this question covers the design behind it.

## Remember
- GPT = decoder-only Transformer with masked self-attention.
- Trained to predict the next token, then tuned to follow instructions.
- Writes one token at a time, using a KV cache for speed.

## Related Questions
- [What is a decoder?](../Week-04-Transformer-Architecture/12-What-is-a-decoder.md)
- [What is masked self-attention?](../Week-04-Transformer-Architecture/13-What-is-masked-self-attention.md)
- [What does GPT stand for?](../Week-02-Introduction-to-LLMs-and-Generative-AI/04-What-does-GPT-stand-for.md)
- [What is the KV cache and why does it speed up generation? (bonus)](../Week-04-Transformer-Architecture/16-BONUS-What-is-the-KV-cache-and-why-does-it-speed-up-generation.md)
