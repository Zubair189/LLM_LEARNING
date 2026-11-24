# How does tokenization work?

> Week 02 – Introduction to LLMs and Generative AI · Question 7

## Short Answer
Tokenization converts raw text into a sequence of integer token IDs the model can process, and converts IDs back into text afterwards. Modern LLMs use **subword** algorithms, most commonly **Byte-Pair Encoding (BPE)**, and also WordPiece or Unigram/SentencePiece. These learn a vocabulary from a large corpus by repeatedly **merging the most frequent adjacent symbol pairs**. At encoding time, the text is split into pieces from that learned vocabulary. Frequent words become single tokens; rare words break into several known pieces. Because the tokenizer works at the byte level, any input can be encoded and nothing is "unknown".

## Detailed Explanation
**The full pipeline:**
```
"Hello world!"
   │ 1. normalize       (Unicode normalization; some tokenizers lowercase, most LLMs don't)
   │ 2. pre-tokenize    (split on spaces/punctuation with a regex: "Hello", " world", "!")
   │ 3. apply subword model (BPE merges / Unigram) → pieces
   ▼
["Hello", " world", "!"]
   │ 4. map pieces → IDs using the vocabulary
   ▼
[15496, 995, 0]  ──►  embedding lookup ──► Transformer
                         ...model output IDs ──► 5. decode IDs back to text
```

**BPE training (learning the vocabulary):**
1. Start with a base vocabulary of single characters (or all 256 bytes, in *byte-level* BPE).
2. Count every adjacent pair of symbols in the corpus.
3. Merge the most frequent pair into a new symbol, for example `t` + `h` → `th`.
4. Repeat until the vocabulary reaches the target size (e.g. 100k).
5. Save the **ordered merge list** and the **vocabulary**.

**BPE encoding (using the vocabulary):** apply the learned merges, in the order they were learned, to the new text.

**Algorithm comparison:**
| Algorithm | Used by (examples) | Idea |
|---|---|---|
| Byte-level BPE | GPT family, Llama 3+, many modern LLMs | Merge frequent byte pairs |
| WordPiece | BERT | Merge pairs that most increase the likelihood of the data |
| Unigram (SentencePiece) | T5, some multilingual models | Start large, prune pieces that matter least |

**Practical consequences:**
- A model and its tokenizer are a **matched pair**. You can't swap tokenizers without retraining.
- Languages that were under-represented in the tokenizer's training data split into more tokens, so they cost more and fit less into the context.
- Odd tokenization of numbers (e.g. `"1234"` → `"123"`, `"4"`) is one reason LLMs used to make arithmetic slips.

> **Note on overlap:** see [What is a token?](./06-What-is-a-token.md) for the definition and [Why tokens instead of words?](./08-Why-do-LLMs-use-tokens-instead-of-complete-words.md) for the design rationale. This question is about the **algorithm**.

## Example
**Analogy:** Making up abbreviations while taking notes. If you write "the" constantly, you invent a shorthand for it. Later "th" + "ing" becomes another shorthand. BPE does exactly this: the most frequent combinations earn their own symbol.

**Python: a minimal BPE trainer and encoder (standard library):**
```python
from collections import Counter

def train_bpe(words, num_merges):
    # each word = tuple of characters + end-of-word marker
    vocab = Counter(tuple(w) + ("</w>",) for w in words)
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for word, freq in vocab.items():
            for a, b in zip(word, word[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        new_vocab = Counter()
        for word, freq in vocab.items():
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    out.append(word[i] + word[i + 1]); i += 2
                else:
                    out.append(word[i]); i += 1
            new_vocab[tuple(out)] += freq
        vocab = new_vocab
    return merges

def encode(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:                       # apply merges in learned order
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols[i:i + 2] = [a + b]
            else:
                i += 1
    return symbols

corpus = "low lower lowest newer newest wider widest".split() * 5
merges = train_bpe(corpus, 10)
print(merges[:4])               # [('w', 'e'), ('l', 'o'), ('r', '</w>'), ('s', 't')]
print(encode("lowest", merges)) # ['lowe', 'st</w>']  <- frequent word -> few tokens
print(encode("slowly", merges)) # ['s', 'lo', 'w', 'l', 'y', '</w>']  <- unseen word -> pieces
```
For real tokenizers, use a library: `tiktoken`, Hugging Face `tokenizers`/`transformers` (`AutoTokenizer.from_pretrained(...)`), or the provider's token-counting endpoint.

## Common Mistakes / Misconceptions
- **"Tokenizers split on spaces."** They split into learned subword pieces, and spaces are often attached to the following word.
- **Using the wrong tokenizer** to count tokens for a model. Counts and IDs differ between model families.
- **"The model learns the tokenizer during training."** The tokenizer is trained *separately, beforehand* and then frozen.
- **"Unknown words break the model."** Byte-level BPE can always fall back to bytes.

## Key Takeaways
- Tokenization means normalize → pre-tokenize → apply subword merges → map to IDs (and decode in reverse).
- BPE learns merges of frequent adjacent pairs until it reaches the target vocabulary size.
- Byte-level fallback means any input can be encoded.
- Each model has its own tokenizer, which affects cost, context use and multilingual fairness.

## Related Questions
- [What is a token?](./06-What-is-a-token.md)
- [Why do LLMs use tokens instead of complete words?](./08-Why-do-LLMs-use-tokens-instead-of-complete-words.md)
- [What are token embeddings?](../Week-04-Transformer-Architecture/10-What-are-token-embeddings.md)
- [What is a context window?](./09-What-is-a-context-window.md)
