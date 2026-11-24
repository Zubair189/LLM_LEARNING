# What is a pretrained model?

> Week 06 – Working with LLMs and Model APIs · Question 1

> Note: "Pretrained" is about **how** a model was made, while a "foundation model" ([Q2](02-What-is-a-foundation-model.md)) is a very large, general pretrained model that many apps build on.

## In One Line
A pretrained model is a model that **has already been trained** on lots of data, so you can use it straight away instead of training your own.

## Simple Explanation
Training a model means letting it learn from data. The result of that learning is stored as **weights**: millions or billions of numbers inside the model.

Training a big LLM from zero costs millions of dollars. So a company or lab does it **once**, then shares or sells access to the finished model. That finished model is "pretrained".

For LLMs, **pretraining** means reading a huge amount of text and practising one thing: guessing the next word (more exactly, the next **token**, a small piece of text).

What you can do with a pretrained model:

1. **Use it as it is**, just by writing prompts.
2. **Fine-tune it**: train it a little more on your own small dataset so it learns your style or topic. Reusing learned knowledge like this is called **transfer learning**.

Two kinds you will see:
- **Base model:** only continues text. Not good at following orders.
- **Instruct / chat model:** trained extra to follow instructions. Names often end in `-Instruct`.

**Link to LLMs:** every LLM you call, like Claude or Llama, is a pretrained model.

## Real-Life Example
Hiring a graduate who already speaks the language and knows general facts, then giving them a short company training, is much faster than raising a child from birth to do the job. The graduate is the pretrained model; the company training is fine-tuning.

## Code Example
```python
# pip install transformers torch
from transformers import pipeline                     # easy way to use Hugging Face models

classifier = pipeline("sentiment-analysis",           # the task we want
    model="distilbert-base-uncased-finetuned-sst-2-english")  # a pretrained model
print(classifier("The answer was clear and helpful."))  # use it, no training

# Output (e.g., score varies slightly):
# [{'label': 'POSITIVE', 'score': 0.99...}]
```

**What happened:** we downloaded someone else's already-trained model and used it in two lines. No training happened on our side.

## Watch Out
- A pretrained model only knows things up to its **training cutoff date**. For fresh facts, give it the information in the prompt (RAG).
- Don't use a base model as a chatbot. Pick the instruct/chat version.

## Remember
- Pretrained = **already trained**, ready to use.
- LLMs are pretrained by **guessing the next token** on huge text.
- You can **prompt** it or **fine-tune** it.

## Related Questions
- [What is a foundation model?](02-What-is-a-foundation-model.md)
- [What is Hugging Face?](03-What-is-Hugging-Face.md)
- [What is a model repository?](04-What-is-a-model-repository.md)
- [What is prompt engineering?](09-What-is-prompt-engineering.md)
