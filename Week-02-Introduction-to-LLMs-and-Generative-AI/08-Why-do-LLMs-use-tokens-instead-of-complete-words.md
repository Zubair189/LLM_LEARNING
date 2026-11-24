# Why do LLMs use tokens instead of complete words?

> Week 02 – Introduction to LLMs and Generative AI · Question 8

## Short Answer
Subword tokens are a compromise between using whole words and using single characters. A whole-word vocabulary would be enormous and still couldn't handle new words, typos, names or code ("out-of-vocabulary" failures). Characters would make sequences very long and slow to process. Subword tokens keep the vocabulary at a manageable size (around 50k–250k entries), let the model represent **any** string by combining pieces, share meaning across related words ("play", "playing", "played"), and keep sequences reasonably short. That makes training and inference efficient.

## Detailed Explanation
**The three options:**
| Unit | Vocabulary size | Sequence length | Handles new words / typos? | Problem |
|---|---|---|---|---|
| Whole words | Millions (every word form in every language, names, code identifiers...) | Short | **No**; unknown words become `<UNK>` | Huge embedding table; rare words poorly learned |
| Characters / bytes | ~100–256 | **Very long** (4–5× more units) | Yes | Slow and costly; attention cost grows with length; harder to learn meaning |
| **Subword tokens** | ~50k–250k | Moderate | **Yes** (falls back to smaller pieces) | Some quirks (spelling, numbers) |

**Specific benefits of subwords:**
- **No out-of-vocabulary problem.** `"Claudeify"` → `"Claude" + "ify"`. Byte-level fallback covers emoji and any script.
- **Morphology and sharing.** `"unhappiness"` → `"un" + "happi" + "ness"`, so the model reuses what it knows about each piece.
- **Efficiency.** Frequent words get one token, so common text stays short, which saves compute and context space.
- **Multilingual and code support.** One vocabulary can cover many languages and programming syntax.
- **A manageable output layer.** At every step the model computes a probability for *every* vocabulary entry. With millions of entries that would be very expensive.

**Why sequence length matters so much:** self-attention compares tokens with each other, so cost grows roughly with the **square** of sequence length (Week 4). Making sequences 4× longer by using characters would mean roughly 16× the attention compute.

**Trade-offs and quirks of tokens:**
- Character-level tasks (counting letters, reversing strings, rhymes) are harder because the model doesn't directly "see" letters.
- Number splitting is irregular, which hurt arithmetic in older models. Many newer tokenizers split digits more consistently.
- Languages under-represented in the tokenizer's training data need more tokens per word, which means higher cost and less effective context.

Researchers keep exploring byte-level or "tokenizer-free" models, but as of 2026 subword tokenization is still the norm in production LLMs.

> **Note on overlap:** see [What is a token?](./06-What-is-a-token.md) (definition) and [How does tokenization work?](./07-How-does-tokenization-work.md) (the BPE algorithm). This question is the *design rationale*.

## Example
**Analogy:** Written Chinese uses thousands of characters (like a word-level vocabulary: huge, and new words need new characters). English spells everything with 26 letters (character-level: tiny vocabulary, long spellings). Syllable-based systems such as Japanese kana sit in between, with a moderate set that can spell anything compactly. Subword tokens are the "syllables" of LLMs.

**Python: compare vocabulary size and sequence length for words vs. characters:**
```python
text = ("the player played and the players are playing "
        "unplayable games while replaying old plays").split()

word_vocab = set(text)
char_vocab = set("".join(text))
print("word vocab:", len(word_vocab), "| seq len:", len(text))
print("char vocab:", len(char_vocab), "| seq len:", sum(len(w) for w in text))
# word vocab: 13 | seq len: 14
# char vocab: 18 | seq len: 75

# A word-level model has never seen "replayed" -> <UNK>
print("replayed" in word_vocab)   # False -> unknown word!
# A subword model can compose it from known pieces:
pieces = ["re", "play", "ed"]
print("".join(pieces))            # replayed
```
Even in this tiny corpus, words like "play/played/playing/plays/replaying" each need a separate whole-word entry. Subwords share the piece `play` across all of them.

## Common Mistakes / Misconceptions
- **"Tokens are just a cost-accounting unit."** They shape what the model can easily learn and do.
- **"Bigger vocabulary is always better."** It shortens sequences but enlarges the embedding and output layers, and leaves rare tokens under-trained.
- **"Character models would be strictly smarter."** They get exact spelling but pay a large price in efficiency and in modelling long-range meaning.
- **"Tokenization is identical across languages."** Efficiency varies widely by language.

## Key Takeaways
- Whole words: vocabulary too large and can't handle new words. Characters: sequences too long.
- Subwords give a fixed, moderate vocabulary that can still encode any text.
- Shorter sequences matter because attention cost grows roughly quadratically with length.
- Side effects include weaker spelling and arithmetic and uneven cost across languages.

## Related Questions
- [What is a token?](./06-What-is-a-token.md)
- [How does tokenization work?](./07-How-does-tokenization-work.md)
- [What is a context window?](./09-What-is-a-context-window.md)
- [What is self-attention?](../Week-04-Transformer-Architecture/05-What-is-self-attention.md)
