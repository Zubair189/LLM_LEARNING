# Why do LLMs use tokens instead of complete words?

> Week 02 – Introduction to LLMs and Generative AI · Question 8

## In One Line
Word pieces (tokens) are the **best middle ground**: fewer pieces to learn than whole words, but much shorter than spelling everything letter by letter.

## Simple Explanation
There are three ways a model could read text:

1. **Whole words:**
   - The list of words would be **huge** (every word, name, typo and language).
   - New words like "Claudeify" or typos would be **unknown**.
2. **Single letters:**
   - Only a small set of symbols, and nothing is unknown.
   - But text becomes **very long**, about 4–5 times more pieces. Longer input means much more work for the model.
3. **Word pieces (tokens):** the middle ground.
   - A **moderate vocabulary** (around 50,000–250,000 pieces).
   - **Any** word can be built from pieces: "Claudeify" → "Claude" + "ify".
   - Related words **share pieces**: "play", "played", "playing" all reuse "play".
   - Common words are a single token, so text stays **short**.

**Why short matters:** the model's attention step compares tokens with each other, so the work grows very fast as text gets longer. Roughly, 4× longer text means about 16× more attention work.

**The downside:** the model sees pieces, not letters. That is why LLMs can struggle with tasks like counting the letters in a word.

Question 6 explains what a token is, and Question 7 explains how text is split. This question explains **why**.

## Real-Life Example
Writing with whole-word symbols needs thousands of symbols. Writing with 26 letters makes words long. Writing in **syllables** sits in between: a moderate set that can spell anything compactly. Tokens are the "syllables" of LLMs.

## Code Example
Compare whole words and letters on a tiny text.

```python
text = "the player played and the players are playing old plays".split()

word_vocab = set(text)                     # all different words
char_vocab = set("".join(text))            # all different letters

print("words:", len(word_vocab), "different,", len(text), "in a row")
print("chars:", len(char_vocab), "different,", sum(len(w) for w in text), "in a row")
print("replayed" in word_vocab)            # a new word: unknown to a word model!

pieces = ["re", "play", "ed"]              # a token model builds it from known pieces
print("".join(pieces))

# Output:
# words: 9 different, 10 in a row
# chars: 14 different, 46 in a row
# False
# replayed
```

**What happened:** letters made the text almost 5 times longer, and the word list could not handle "replayed". Word pieces solve both problems.

## Watch Out
- **A bigger vocabulary is not always better.** It shortens text but makes the model larger, and rare pieces are poorly learned.
- **Not all languages are equal.** Languages with less training text often need more tokens per word, so they cost more.

## Remember
- Whole words: too many and can't handle new words.
- Letters: text gets too long.
- Tokens (word pieces): the practical middle.

## Related Questions
- [What is a token?](./06-What-is-a-token.md)
- [How does tokenization work?](./07-How-does-tokenization-work.md)
- [What is a context window?](./09-What-is-a-context-window.md)
- [What is self-attention?](../Week-04-Transformer-Architecture/05-What-is-self-attention.md)
