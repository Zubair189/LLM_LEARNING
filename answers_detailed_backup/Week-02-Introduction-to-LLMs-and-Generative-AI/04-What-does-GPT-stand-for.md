# What does GPT stand for?

> Week 02 – Introduction to LLMs and Generative AI · Question 4

## Short Answer
GPT stands for **Generative Pre-trained Transformer**. **Generative** means it produces new text. **Pre-trained** means it is first trained on a huge general text corpus before being adapted to specific uses. **Transformer** is the neural-network architecture it is built on, introduced in the 2017 paper "Attention Is All You Need". GPT is the name of OpenAI's model family (GPT-1 in 2018 through today's models), but the recipe it names, a decoder-only Transformer pretrained on next-token prediction, is shared by almost every modern LLM, including Claude, Gemini, Llama and Mistral.

## Detailed Explanation
| Word | Meaning | Why it matters |
|---|---|---|
| **Generative** | Outputs new content token by token (autoregressive) | Contrasts with models that only classify or encode, such as BERT |
| **Pre-trained** | Learns general language from massive unlabelled text by predicting the next token (self-supervised) | One expensive pretraining run is reused for many tasks via prompting or fine-tuning |
| **Transformer** | Architecture based on **self-attention**, which lets every token look at the relevant earlier tokens | Parallelizes well on GPUs and scales to billions of parameters |

**GPT is a "decoder-only" Transformer:**
```
Original Transformer (2017):  Encoder ──► Decoder      (machine translation)
BERT (2018):                  Encoder only             (understanding / classification)
GPT (2018 → ):                Decoder only             (generation; predicts next token,
                                                        masked so it can't see the future)
```

**A short lineage:**
- **GPT-1 (2018):** showed that "pretrain, then fine-tune" works.
- **GPT-2 (2019):** 1.5B parameters; strong zero-shot generation.
- **GPT-3 (2020):** 175B parameters; *in-context / few-shot learning* from prompts alone.
- **InstructGPT / ChatGPT (2022):** added instruction tuning + RLHF and made LLMs mainstream.
- Since then, many labs (Anthropic, Google, Meta, Mistral, Alibaba, DeepSeek, ...) build GPT-style decoder-only models with their own improvements.

**"GPT" in interviews:** people often say "GPT" to mean "any LLM". It is more precise to say *GPT-style* or *decoder-only Transformer* for the architecture, and to keep "GPT" for OpenAI's product line.

> **Note on overlap:** Week 4's [How does GPT use the Transformer architecture?](../Week-04-Transformer-Architecture/14-How-does-GPT-use-the-Transformer-architecture.md) goes inside the model: masked self-attention, stacked decoder blocks, and the output softmax. This question is only about what the acronym means and where it came from.

## Example
**Analogy:** Think of a chef's training. **Pre-trained**: years at culinary school learning every cuisine (general knowledge). **Generative**: they cook new dishes rather than just judging others' food. **Transformer**: their method, which is to look at all the ingredients at once and decide which ones matter for the next step.

**Python: the "G" in GPT is autoregressive generation, a loop that appends one token at a time:**
```python
# Toy "pre-trained" next-word table standing in for a Transformer
next_word = {
    "<start>": "Attention", "Attention": "is", "is": "all",
    "all": "you", "you": "need", "need": "<end>",
}

tokens = ["<start>"]
while tokens[-1] != "<end>":
    tokens.append(next_word[tokens[-1]])   # predict next token from context
print(" ".join(tokens[1:-1]))  # Attention is all you need
```
A real GPT-style model replaces the lookup table with a Transformer that looks at *all* previous tokens and outputs a probability for every word in its vocabulary.

## Common Mistakes / Misconceptions
- **"Pre-trained means ready for any task out of the box."** A pretrained *base* model just continues text. Chat behaviour comes from later instruction tuning and RLHF.
- **"GPT = ChatGPT."** ChatGPT is a product; GPT is the model family and architecture style.
- **"All LLMs are GPTs."** Most are GPT-*style* (decoder-only), but they are different models from different organizations.
- **Writing "General Purpose Technology" or "Generative Pre-Training"** in an exam. The acronym is Generative **Pre-trained Transformer**.

## Key Takeaways
- GPT = Generative Pre-trained Transformer.
- Generative: produces text; Pre-trained: learned general language first; Transformer: attention-based architecture.
- GPT models are decoder-only, autoregressive Transformers, and most modern LLMs follow this recipe.
- ChatGPT-style behaviour comes from post-training on top of the pretrained model.

## Related Questions
- [How does GPT use the Transformer architecture?](../Week-04-Transformer-Architecture/14-How-does-GPT-use-the-Transformer-architecture.md)
- [What is a Transformer?](../Week-04-Transformer-Architecture/02-What-is-a-Transformer.md)
- [What is a pretrained model?](../Week-06-Working-with-LLMs-and-Model-APIs/01-What-is-a-pretrained-model.md)
- [How are chat models trained (pretraining, instruction tuning, RLHF)?](./15-BONUS-How-are-chat-models-trained-pretraining-instruction-tuning-RLHF.md)
