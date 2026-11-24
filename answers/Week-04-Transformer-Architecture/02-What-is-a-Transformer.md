# What is a Transformer?

> Week 04 – Transformer Architecture · Question 2

## In One Line
A Transformer is the neural-network design behind almost every modern LLM; it lets **every word look at every other word at once** to understand context.

## Simple Explanation
The Transformer was introduced in 2017 in a paper called *"Attention Is All You Need"*. It replaced the older word-by-word RNNs.

**How it works, step by step (GPT-style):**
1. **Split the text into tokens** (word pieces) and turn each one into a list of numbers called an **embedding**.
2. **Add position information**, so the model knows which word came first, second, and so on.
3. **Pass everything through many identical blocks.** Each block has two parts:
   - **Attention:** each word gathers useful information from the other words. For example, "it" looks back to find what "it" refers to.
   - **Feed-forward network:** each word then "thinks" on its own, using a small neural network. This part stores much of the model's knowledge.
4. **Predict the next token:** at the end, the model gives a probability for every possible next token.

Two helpers sit around each part: **residual connections** (add the input back to the output, so information and learning signals flow easily through many layers) and **normalisation** (keeps the numbers in a stable range).

**Three types of Transformer:**
- **Encoder-only** (e.g. BERT): good at understanding text.
- **Encoder–decoder** (e.g. the original 2017 model, T5): good at translation.
- **Decoder-only** (e.g. GPT, Claude, Llama): good at generating text. This is what today's chatbots use.

**Link to LLMs:** GPT, Claude, Gemini and Llama are all Transformers, mostly decoder-only.

## Real-Life Example
Imagine a meeting where everyone can read the whole document at once. In each round, every person asks "who here has information I need?", listens to them, then thinks privately. After many rounds, everyone understands the document well. An RNN would be one person passing notes down a line instead.

## Watch Out
- A Transformer is not "only attention". The feed-forward parts hold most of the parameters, and the helpers (residuals, normalisation) are essential.
- Attention by itself does not know word order; position information must be added.

Note: Week 12 Q2 gives the full revision-level walkthrough; this answer is the first, simple definition.

## Remember
- Transformer = stacked blocks of **attention + feed-forward**.
- All words are processed at once, and any two words can connect directly.
- Modern chatbots use the **decoder-only** version.

## Related Questions
- [What problems do traditional RNNs have?](../Week-04-Transformer-Architecture/01-What-problems-do-traditional-RNNs-have.md)
- [What is self-attention?](../Week-04-Transformer-Architecture/05-What-is-self-attention.md)
- [How does GPT use the Transformer architecture?](../Week-04-Transformer-Architecture/14-How-does-GPT-use-the-Transformer-architecture.md)
- [Explain Transformer architecture.](../Week-12-Final-Project-and-Revision/02-Explain-Transformer-architecture.md)
