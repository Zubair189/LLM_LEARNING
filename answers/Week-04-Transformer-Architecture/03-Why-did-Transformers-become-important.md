# Why did Transformers become important?

> Week 04 – Transformer Architecture · Question 3

## In One Line
Transformers made it possible to **train huge models on huge amounts of text quickly**, and they kept getting better as they got bigger.

## Simple Explanation
Several reasons came together:

1. **Fast, parallel training.** RNNs read one word at a time. Transformers process all words in a training example **at the same time**, using big matrix multiplications. GPUs (graphics cards) are built for exactly this, so training became much faster.
2. **Better with long text.** With attention, any word can connect directly to any other word, even far away. RNNs struggled with this.
3. **Bigger really is better (scaling laws).** Researchers found that Transformers improve in a **smooth, predictable** way when you add more parameters, more data and more computing power. Companies could predict the payoff before spending money, so they built ever-larger models.
4. **One model, many tasks.** A single model **pre-trained** on lots of raw text (learning by predicting the next word) can then be adapted to translation, summarising, coding, chat and more. Before, each task needed its own model.
5. **Works on more than text.** The same design works for images, audio, proteins and more, as long as the input can be cut into tokens.

**A short timeline:**
- **2017:** the Transformer paper.
- **2018:** BERT and GPT-1.
- **2020:** GPT-3 shows it can learn tasks from a few examples in the prompt.
- **2022:** ChatGPT brings LLMs to everyone.
- **Since then:** open models (Llama, Qwen, Mistral, DeepSeek), longer contexts, multimodal and reasoning models.

**Link to LLMs:** every major LLM today is built on (or mostly on) the Transformer.

## Real-Life Example
Before the printing press, books were copied by hand, page after page (like RNNs). The press printed many pages at once (parallel), and more investment gave predictably more books (scaling). It didn't just speed up copying; it changed what was possible.

## Watch Out
- Transformers often do **more** total arithmetic than RNNs. They won because that work can be done **in parallel**, not because it is less.
- Training is parallel, but **generating** text is still one token at a time.

Note: this question is about the historical impact; Q15 explains why the design suits language, and Q2 defines it.

## Remember
- Parallel training on GPUs made huge models possible.
- Attention handles long-range links well.
- Predictable scaling and one-model-many-tasks drove the AI boom.

## Related Questions
- [What problems do traditional RNNs have?](../Week-04-Transformer-Architecture/01-What-problems-do-traditional-RNNs-have.md)
- [What is a Transformer?](../Week-04-Transformer-Architecture/02-What-is-a-Transformer.md)
- [Why are Transformers effective at processing language?](../Week-04-Transformer-Architecture/15-Why-are-Transformers-effective-at-processing-language.md)
- [What is model pretraining?](../Week-09-Fine-tuning-and-Model-Optimization/01-What-is-model-pretraining.md)
