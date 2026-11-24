# What is a Large Language Model (LLM)?

> Week 02 – Introduction to LLMs and Generative AI · Question 2

## Short Answer
A Large Language Model is a deep neural network, almost always a **Transformer**, trained on enormous amounts of text (trillions of tokens) to predict the next token. "Large" refers to its billions of parameters and the scale of its training data. Next-token prediction at that scale produces broad abilities: answering questions, writing, summarizing, translating, coding and reasoning. After pretraining, LLMs are tuned with instruction data and human or AI feedback, which turns them into helpful chat assistants such as Claude, GPT, Gemini, Llama, Mistral and Qwen.

## Detailed Explanation
**Breaking down the name:**
- **Large:** billions to trillions of parameters, trained on trillions of tokens with huge GPU/TPU clusters.
- **Language:** works on human language and code. Modern LLMs are often **multimodal** and also accept images, PDFs or audio.
- **Model:** a learned mathematical function, `tokens in → probability distribution over the next token`.

**How an LLM produces text:**
```
"The capital of France is"  ──tokenize──►  [464, 3139, 286, 4881, 318]
          │
          ▼
  Transformer (N layers of attention + feed-forward, billions of weights)
          │
          ▼
  probabilities: " Paris" 0.92, " Lyon" 0.01, ...  ──sample──►  " Paris"
          │
          └── append token, repeat until a stop token or max_tokens
```
(The token IDs are illustrative. Each tokenizer has its own vocabulary.)

**Lifecycle in three stages** (details in the bonus question):
1. **Pretraining:** next-token prediction on web, books and code produces a *base model* that is knowledgeable but not yet an assistant.
2. **Post-training:** supervised instruction tuning plus RLHF or other RL makes it a helpful, safe *chat model*.
3. **Inference:** you send prompts through an API or run the weights locally.

**Abilities that emerge from scale:** in-context learning (learning from examples in the prompt), following instructions, tool use, and step-by-step reasoning.

**Important limitations:**
- A **knowledge cutoff**: it doesn't know about events after its training data ends, unless you give it tools or retrieval (RAG).
- **Hallucinations**: confident but wrong output.
- A **finite context window** and a cost per token.
- It has **no memory between API calls**. You resend the conversation each time.

> **Note on overlap:** Q5 ([What is a language model?](./05-What-is-a-language-model.md)) covers the *general concept*: any model that assigns probabilities to word sequences, including tiny n-gram models. This question is about the *large, Transformer-based, instruction-tuned* systems built from that concept.

## Example
**Analogy:** Imagine someone who has read a large share of the public internet and has extremely good autocomplete instincts. Ask them anything and they continue the text in the most plausible way. They are usually right and articulate, and occasionally confidently wrong. They also have no idea what happened after the day they stopped reading.

**Python: calling an LLM through the Anthropic SDK:**
```python
import anthropic

client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
resp = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=150,
    system="You are a concise tutor for beginners.",
    messages=[{"role": "user", "content": "In one sentence, what is an LLM?"}],
)
print(resp.content[0].text)
print(resp.usage.input_tokens, resp.usage.output_tokens)  # billing is per token
# e.g. "An LLM is a neural network trained on vast text to predict and generate language."
```
Open-weight LLMs (Llama, Mistral, Qwen, Gemma, ...) can be run locally with tools like Ollama or Hugging Face `transformers` (Week 6).

## Common Mistakes / Misconceptions
- **"LLMs look things up on the internet."** Not by default. They generate from their learned weights unless given a search or RAG tool.
- **"LLMs remember our previous chats."** The model itself is stateless; the application resends the history.
- **"Bigger is always better."** Smaller, cheaper models often suit focused tasks. Choose by quality, latency and cost.
- **"It's just autocomplete, so it can't reason."** Next-token prediction at scale, plus RL post-training, produces real multi-step problem solving. It is still imperfect.

## Key Takeaways
- An LLM is a large Transformer trained to predict the next token on massive text data.
- Pretraining gives knowledge; post-training (SFT + RLHF/RL) gives helpful assistant behaviour.
- It generates text token by token by sampling from a probability distribution.
- Limits: knowledge cutoff, hallucinations, context window, statelessness and cost.

## Related Questions
- [What is a language model?](./05-What-is-a-language-model.md)
- [What is a token?](./06-What-is-a-token.md)
- [How are chat models trained (pretraining, instruction tuning, RLHF)?](./15-BONUS-How-are-chat-models-trained-pretraining-instruction-tuning-RLHF.md)
- [What is a Transformer?](../Week-04-Transformer-Architecture/02-What-is-a-Transformer.md)
