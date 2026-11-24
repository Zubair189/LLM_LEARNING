# What is an encoder?

> Week 04 – Transformer Architecture · Question 11

## In One Line
An encoder is the part of a Transformer that **reads the whole input at once** and turns every word into a context-aware list of numbers, for **understanding** text rather than writing it.

## Simple Explanation
An encoder is a stack of identical blocks. Each block has:
1. **Self-attention with no mask:** every word can look at **every** other word, both before and after it. This is called **bidirectional** attention.
2. **A feed-forward network:** each word then gets processed on its own.

The output is one vector per input word, and each vector now "knows" its context.

**Example:** in "I went to the **bank** to deposit money", the encoder's vector for "bank" can use "deposit money", which comes **after** it. So it knows this is a money bank, not a river bank.

**How encoders are trained:** not by guessing the next word (that would be too easy when you can see the whole sentence). Instead, about 15% of words are hidden and the model must fill them in: "The [MASK] sat on the mat" → "cat". This is called **masked language modelling**.

**What encoders are used for:**
- **Classifying text**, e.g. spam or not spam, positive or negative review.
- **Making embeddings for search** (RAG): average the word vectors into one vector for the whole sentence.
- **Feeding a decoder** in translation models, where the decoder looks at the encoder's output.

**Famous example:** BERT is an encoder-only model.

**Link to LLMs:** chatbots like GPT and Claude are decoder-only, so they don't use a separate encoder. But encoders are still widely used for search, ranking results and fast classification.

## Real-Life Example
A reader who reads the whole paragraph first, and only then writes notes about each word. Because they've seen everything, each note uses the full context. A decoder is like a reader who must write a note on each word before seeing what comes next.

## Watch Out
- Encoders do **not** generate text on their own. For writing, you need a decoder.
- Encoders are not old or useless; they are still the best cheap choice for many search and classification tasks.

## Remember
- Encoder = reads everything at once, words see both sides.
- Trained by filling in hidden words.
- Used for understanding: classification, search embeddings, translation input.

## Related Questions
- [What is a decoder?](../Week-04-Transformer-Architecture/12-What-is-a-decoder.md)
- [What is self-attention?](../Week-04-Transformer-Architecture/05-What-is-self-attention.md)
- [What is masked self-attention?](../Week-04-Transformer-Architecture/13-What-is-masked-self-attention.md)
- [What is the difference between word and sentence embeddings?](../Week-07-Embeddings-and-Vector-Databases/05-What-is-the-difference-between-word-and-sentence-embeddings.md)
