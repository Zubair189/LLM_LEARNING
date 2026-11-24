# What does GPT stand for?

> Week 02 – Introduction to LLMs and Generative AI · Question 4

## In One Line
GPT stands for **Generative Pre-trained Transformer**: a model that **generates** text, was **pre-trained** on lots of text first, and is built with the **Transformer** design.

## Simple Explanation
Each word tells you something:

1. **Generative:** it **creates** new text, one piece (token) at a time.
2. **Pre-trained:** before it is used for anything specific, it first learns general language by reading a huge amount of text and guessing the next word. This one big learning step can then be reused for many tasks.
3. **Transformer:** the type of neural network it uses, introduced in 2017 in a paper called "Attention Is All You Need". Its key trick, called **attention**, lets each word look back at the other words that matter most.

**A short history:**
- **GPT-1 (2018):** showed that "pre-train first, then adapt" works.
- **GPT-3 (2020):** very large; could do new tasks just from examples in the prompt.
- **ChatGPT (2022):** added extra training to follow instructions and chat, which made LLMs famous.

"GPT" is the name of OpenAI's model family. But the same recipe is used by almost every modern LLM, including Claude, Gemini and Llama. So people often say "GPT-style model" to mean "this kind of LLM".

Week 4 has a question on how GPT uses the Transformer inside. This one is only about the name.

## Real-Life Example
Think of a chef. **Pre-trained:** years at cooking school learning every cuisine. **Generative:** they cook new dishes, not just judge other people's food. **Transformer:** their method of looking at all the ingredients at once and deciding which matter for the next step.

## Code Example
The "Generative" part is a simple loop: predict the next word, add it, repeat.

```python
next_word = {                                  # a tiny "pre-trained" lookup table
    "<start>": "Attention", "Attention": "is", "is": "all",
    "all": "you", "you": "need", "need": "<end>",
}

words = ["<start>"]
while words[-1] != "<end>":                    # keep going until the end marker
    words.append(next_word[words[-1]])         # predict the next word and add it

print(" ".join(words[1:-1]))                   # drop the start/end markers
# Output:
# Attention is all you need
```

**What happened:** the text was built one word at a time, each chosen from what came before. A real GPT replaces the lookup table with a Transformer that looks at **all** previous words.

## Watch Out
- **GPT is not the same as ChatGPT.** ChatGPT is a product; GPT is the model family.
- **"Pre-trained" doesn't mean "ready to chat".** A pre-trained model just continues text. Chat skills come from extra training afterwards.

## Remember
- GPT = **G**enerative **P**re-trained **T**ransformer.
- Most modern LLMs follow the same GPT-style recipe.
- Chat behaviour comes from extra training after pre-training.

## Related Questions
- [How does GPT use the Transformer architecture?](../Week-04-Transformer-Architecture/14-How-does-GPT-use-the-Transformer-architecture.md)
- [What is a Transformer?](../Week-04-Transformer-Architecture/02-What-is-a-Transformer.md)
- [What is a pretrained model?](../Week-06-Working-with-LLMs-and-Model-APIs/01-What-is-a-pretrained-model.md)
- [How are chat models trained (pretraining, instruction tuning, RLHF)?](./15-BONUS-How-are-chat-models-trained-pretraining-instruction-tuning-RLHF.md)
