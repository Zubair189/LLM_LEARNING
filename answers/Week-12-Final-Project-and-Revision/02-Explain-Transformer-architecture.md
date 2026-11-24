# Explain Transformer architecture.

> Week 12 – Final Project and Revision · Question 2

## In One Line
A Transformer is the neural network design behind LLMs: it lets every word **look at the other words around it** (attention) to understand meaning, then predicts the next word.

## Simple Explanation
Older models (**RNNs**) read text one word at a time, like reading through a straw. They were slow and forgot the start of long sentences. The Transformer (2017) fixed this: it looks at **all the words at once** and lets each word decide which other words matter to it.

Here is how a GPT-style model goes from text to the next word:

```
"The cat sat" → Tokens → Numbers (embeddings + position) → [Attention → Feed-forward] × many layers → Next-word scores → "on"
```

**Step by step:**

1. **Tokenize.** The text is split into tokens, and each token gets an ID number.
2. **Embed.** Each ID becomes a list of numbers (a **vector**) that captures its meaning. Similar words get similar vectors.
3. **Add position.** Attention on its own doesn't know word order, so we add information about *where* each word is. Without this, "dog bites man" and "man bites dog" would look the same.
4. **Self-attention.** Each word asks: "Which other words should I pay attention to?" To do this, each word makes three things:
   - a **Query**: what am I looking for?
   - a **Key**: what do I offer?
   - a **Value**: my actual content.
   Comparing one word's Query with every word's Key gives attention scores. **Softmax** turns these into percentages, and the word collects a mix of the other words' Values. For example, in "the animal didn't cross the street because **it** was tired", attention helps "it" focus on "animal".
5. **Multi-head attention.** Several attention "heads" run side by side. One might track grammar, another who-did-what.
6. **Feed-forward network.** A small neural network then processes each word on its own. Much of the model's stored knowledge lives here.
7. **Repeat.** Steps 4–6 form one **layer** (or "block"). Big models stack dozens of them. Extra connections and normalisation keep training stable.
8. **Predict.** The final layer gives a score for every token in the vocabulary. Softmax turns them into probabilities, and the model picks the next token, e.g. "on".

**The causal mask:** in GPT-style (**decoder-only**) models, each word may only look at words **before** it, never after. That matches the job of predicting the next word.

**Other versions:** **encoder-only** models (like BERT) look both ways and are good for understanding and embeddings. **Encoder-decoder** models (like the original 2017 design) are good for translation.

**Why Transformers won:** they train in parallel on GPUs, connect far-apart words easily, and keep improving as they get bigger. **The cost:** attention compares every word with every other word, so work grows quickly with text length. That is one reason context windows have a limit.

## Real-Life Example
A meeting where everyone can hear everyone at once. For each topic, a person asks a question (Query), checks everyone's name tag (Key), and takes notes from the most relevant people (Value). Several note-takers each focus on something different (multi-head). In a GPT-style meeting, you can only listen to people who already spoke (causal mask).

## Watch Out
- Attention is not memory of past chats. It only works on the tokens in the **current** prompt.
- Training happens in parallel; only **generating** text goes one token at a time.

## Remember
- Transformer = embeddings + position → many layers of **attention + feed-forward** → next-token probabilities.
- Attention lets each word focus on the words that matter to it.
- Chat LLMs are **decoder-only** with a **causal mask**.

## Related Questions
- [What is a Transformer?](../Week-04-Transformer-Architecture/02-What-is-a-Transformer.md)
- [What are Query, Key and Value?](../Week-04-Transformer-Architecture/06-What-are-Query-Key-and-Value.md)
- [What is multi-head attention?](../Week-04-Transformer-Architecture/08-What-is-multi-head-attention.md)
- [What is masked self-attention?](../Week-04-Transformer-Architecture/13-What-is-masked-self-attention.md)
- [What problems do traditional RNNs have?](../Week-04-Transformer-Architecture/01-What-problems-do-traditional-RNNs-have.md)
- [What is softmax?](../Week-03-Mathematics-and-Neural-Network-Foundations/10-What-is-softmax.md)
