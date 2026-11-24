# What is a token?

> Week 02 – Introduction to LLMs and Generative AI · Question 6

## Short Answer
A token is the basic unit of text an LLM reads and writes. It is usually a **word or a piece of a word** ("un", "believ", "able"), but it can also be punctuation, a space-plus-word, a single character or byte, or a special control marker. Each token maps to an integer ID in the model's vocabulary, which typically holds 50k–250k entries. Everything about using an LLM is measured in tokens: the context window, the `max_tokens` output limit, speed, and **API pricing**. As a rough rule for English, 1 token ≈ 4 characters ≈ ¾ of a word.

## Detailed Explanation
**What tokens look like (illustrative; each tokenizer differs):**
```
Text:    "Tokenization isn't hard!"
Tokens:  ["Token", "ization", " isn", "'t", " hard", "!"]
IDs:     [ 3963,    2065,     2125,  470,  1327,   0 ]   (made-up IDs)
```
Notes:
- The leading space is often **part of** the token (`" hard"`).
- Common words are usually one token; rare words split into several pieces.
- Numbers, code and non-English text often use **more tokens per word**.

**Types of tokens:**
| Kind | Example | Purpose |
|---|---|---|
| Word / subword | `"cat"`, `"ization"` | Normal text |
| Byte-level fallback | Individual UTF-8 bytes | Lets any string (emoji, rare scripts) be encoded |
| Special / control | `<bos>`, `<eos>`, role markers | Mark the start/end of text and chat turns. The API adds these for you |

**Why engineers care about tokens:**
- **Cost:** APIs charge per input token and per output token. Output tokens usually cost several times more than input tokens.
- **Limits:** the prompt, the history, retrieved documents *and* the answer must all fit in the **context window**.
- **Latency:** output is generated one token at a time, so more output tokens means a slower response.
- **Quirks:** models "see" tokens, not letters. That is why older models struggled to count the r's in "strawberry" or to reverse strings.

**Rules of thumb (English):** 100 tokens ≈ 75 words. 1,000 words ≈ 1,300–1,400 tokens. These vary by model and language, so measure with the provider's token counter when it matters.

> **Note on overlap:** this question defines *what* a token is. [How does tokenization work?](./07-How-does-tokenization-work.md) explains *how* text is split (BPE and its relatives). [Why do LLMs use tokens instead of complete words?](./08-Why-do-LLMs-use-tokens-instead-of-complete-words.md) explains *why* subword units are the design choice. Week 11's [token consumption](../Week-11-Production-LLM-Applications/11-What-is-token-consumption.md) covers measuring and budgeting tokens in production.

## Example
**Analogy:** Tokens are like Lego bricks for text. Common shapes (frequent words) come as single big bricks. Unusual shapes (rare words) are built from several smaller bricks. The model only ever handles bricks, never the "word" as a whole.

**Python: estimate tokens and cost with the rough 4-characters rule:**
```python
def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))   # rough English heuristic

prompt = "Summarize the following customer review in one sentence: " + "great " * 200
n_in = estimate_tokens(prompt)
n_out = 50
price_in, price_out = 3.00, 15.00         # example $ per 1M tokens - check current pricing
cost = n_in / 1e6 * price_in + n_out / 1e6 * price_out
print(n_in, f"${cost:.6f}")               # 314 $0.001692
```

**Exact count using the provider's tokenizer (Anthropic SDK):**
```python
import anthropic
client = anthropic.Anthropic()
count = client.messages.count_tokens(
    model="claude-sonnet-5-5",
    messages=[{"role": "user", "content": "Tokenization isn't hard!"}],
)
print(count.input_tokens)   # exact input-token count for this model (includes chat formatting)
```

## Common Mistakes / Misconceptions
- **"1 token = 1 word."** On average a token is about ¾ of a word in English, and it varies a lot.
- **"Token counts are the same across models."** Each model family has its own tokenizer, so the same text gives different counts.
- **Forgetting output tokens and history** when budgeting cost or context. Every turn resends the whole conversation.
- **Assuming the model sees characters.** Spelling, counting letters and character-level tasks can be unreliable because of this.

## Key Takeaways
- A token is a word, subword, byte or special marker, mapped to an integer ID.
- Rough English rule: 1 token ≈ 4 characters ≈ 0.75 words.
- Context limits, latency and pricing are all measured in tokens.
- Tokenizers are model-specific, so use the provider's counter for exact numbers.

## Related Questions
- [How does tokenization work?](./07-How-does-tokenization-work.md)
- [Why do LLMs use tokens instead of complete words?](./08-Why-do-LLMs-use-tokens-instead-of-complete-words.md)
- [What is a context window?](./09-What-is-a-context-window.md)
- [What is token consumption?](../Week-11-Production-LLM-Applications/11-What-is-token-consumption.md)
