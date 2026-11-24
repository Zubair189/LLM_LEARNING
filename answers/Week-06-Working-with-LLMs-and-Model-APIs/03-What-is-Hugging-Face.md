# What is Hugging Face?

> Week 06 – Working with LLMs and Model APIs · Question 3

## In One Line
Hugging Face is a website and set of Python tools where people **share, find and use AI models and datasets**.

## Simple Explanation
Hugging Face is often called **"GitHub for AI"**. It is a company that runs a huge free online hub.

What you find there:

1. **Models:** hundreds of thousands of open models, like Llama, Qwen and Mistral, ready to download.
2. **Datasets:** data for training and testing models.
3. **Spaces:** small demo apps where you can try a model in your browser.
4. **Python libraries:** the most famous is **`transformers`**, which loads and runs models in a few lines.

Every model has a **model card**: a page explaining what the model does, its licence (what you are allowed to do with it), its limits and how much memory it needs.

How it works when you use a model in code:
- The first time, the model files are **downloaded** and saved on your computer (cached).
- After that, it loads from your computer, so it is faster.

Some models are **gated**: you must accept their licence on the website and log in with an access token first.

**Link to LLMs:** it is the main place to find **open** LLMs you can run yourself or fine-tune.

## Real-Life Example
Hugging Face is like an app store combined with a library. You browse models, read the reviews and instructions (model cards), and install one with a single line of code.

## Code Example
```python
# pip install transformers torch
from transformers import pipeline                         # Hugging Face's easy tool

chat = pipeline("text-generation",                        # task: write text
                model="Qwen/Qwen2.5-0.5B-Instruct")       # a small open model from the Hub
messages = [{"role": "user", "content": "Name one use of embeddings."}]
out = chat(messages, max_new_tokens=40)                   # run the model locally
print(out[0]["generated_text"][-1]["content"])            # the model's reply

# Output (e.g.):
# Embeddings are used for semantic search, finding text with similar meaning.
```

**What happened:** the code downloaded a small chat model from the Hugging Face Hub and ran it on our own computer.

## Watch Out
- Hugging Face is a **platform**, not a model. The models come from many companies and people.
- Check the model size before downloading. A huge model may not fit on a laptop.

## Remember
- Hugging Face = **hub of models, datasets and demos**.
- `transformers` lets you **use a model in a few lines**.
- Always **read the model card** (licence, limits, size).

## Related Questions
- [What is a model repository?](04-What-is-a-model-repository.md)
- [What is a pretrained model?](01-What-is-a-pretrained-model.md)
- [What is Ollama?](05-What-is-Ollama.md)
- [What are modules and packages?](../Week-05-Python-for-LLM-Engineering/07-What-are-modules-and-packages.md)
