# What is a foundation model?

> Week 06 – Working with LLMs and Model APIs · Question 2

> Note: Compare with [What is a pretrained model?](01-What-is-a-pretrained-model.md): "pretrained" is about how a model was trained, while "foundation" means it is big and general enough for many products to be built on it.

## In One Line
A foundation model is a **very large, general-purpose model** that can be used for many different tasks just by changing the instructions.

## Simple Explanation
In the past, teams trained a separate model for each job: one for spam, one for translation, one for summaries.

A foundation model replaces many of those with **one big model**. It was trained on a huge, varied mix of data (text, code, and often images), so it can do lots of things.

What makes a model a "foundation model":

1. **Big:** billions of **parameters** (the learned numbers inside the model).
2. **General:** not built for just one task.
3. **Adaptable:** you steer it with prompts, examples, tools, extra documents, or fine-tuning.

Examples: Anthropic's Claude, OpenAI's GPT, Google's Gemini, and open-weight models like Llama, Qwen and Mistral.

You reach them in two ways:
- **Closed models** (like Claude): used through an online API.
- **Open-weight models** (like Llama): you can download them and run them yourself, for example with Ollama.

**Link to LLMs:** most LLMs you use are foundation models. But foundation models can also work with images or audio, not only text.

## Real-Life Example
Think of the foundation of a building. One strong base can hold up a house, a shop or an office, depending on what you build on top. A foundation model is that base, and your app is what you build on it.

## Code Example
One model, three different jobs. Only the prompt changes.

```python
import anthropic                                   # pip install anthropic

client = anthropic.Anthropic()                     # reads ANTHROPIC_API_KEY
tasks = ["Translate to French: Good morning",      # job 1: translate
         "One word, positive or negative? 'Terrible battery.'",  # job 2: classify
         "Python one-liner to reverse a string s."]              # job 3: code

for prompt in tasks:
    resp = client.messages.create(model="claude-sonnet-5-5", max_tokens=1024,
                                  messages=[{"role": "user", "content": prompt}])
    print(resp.content[0].text)                    # print each answer

# Output (e.g.):
# Bonjour
# Negative
# s[::-1]
```

**What happened:** the same model translated, classified and wrote code. We never trained anything; we only changed the instructions.

## Watch Out
- Bigger is not always better. A smaller, cheaper model is often good enough and faster.
- "Open weights" does not always mean "free for any use". Check the licence.

## Remember
- Foundation model = **large, general, adaptable**.
- **One model, many tasks**, steered by prompts.
- Used via an **API** (closed) or **downloaded** (open weights).

## Related Questions
- [What is a pretrained model?](01-What-is-a-pretrained-model.md)
- [What is the difference between local and cloud LLMs?](06-What-is-the-difference-between-local-and-cloud-LLMs.md)
- [What is an inference API?](07-What-is-an-inference-API.md)
- [What is few-shot prompting?](18-BONUS-What-is-few-shot-prompting.md)
