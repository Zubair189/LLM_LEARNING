# Why is Python widely used in AI development?

> Week 05 – Python for LLM Engineering · Question 1

## In One Line
Python is popular in AI because it is **easy to read and write** and it has **ready-made tools for almost every AI job**.

## Simple Explanation
Python code looks close to plain English. A short Python program can do what takes many lines in other languages, so people can try ideas quickly.

Python also has a huge collection of **libraries**. A library is code someone else already wrote that you can reuse. For AI there are libraries for:

1. **Working with data:** NumPy (fast number lists) and Pandas (tables).
2. **Training models:** PyTorch.
3. **Using ready-made models:** Hugging Face `transformers`.
4. **Talking to LLMs:** official kits (SDKs) such as `anthropic`.

**"But isn't Python slow?"** Plain Python loops are slow. But the heavy maths inside these libraries is written in fast languages like C, C++ and CUDA (for GPUs). Python just gives the orders. That is why people call Python a **"glue" language**.

Most researchers also share their code in Python, so new AI ideas usually appear in Python first.

**Link to LLMs:** most LLM apps spend their time waiting for answers over the internet, not doing heavy maths, so Python's speed is rarely the problem.

## Real-Life Example
Think of a film director. The director does not hold the camera or build the sets; skilled crew members do that. The director just tells everyone what to do in simple words. Python is the director, and the fast C/CUDA code is the crew.

## Code Example
A whole LLM call in a few readable lines.

```python
import anthropic                        # the official Claude SDK

client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY from your environment

resp = client.messages.create(
    model="claude-sonnet-5-5",          # which model to use
    max_tokens=1024,                    # longest answer allowed
    system="Reply with one word: positive or negative.",  # instructions
    messages=[{"role": "user", "content": "Great battery life!"}],
)
print(resp.content[0].text)             # print the model's reply

# Output (e.g.):
# positive
```

**What happened:** Python sent a review to Claude and printed the answer. The SDK handled the internet request, the login key and the JSON for us.

## Watch Out
- "Python is slow, so AI is slow" is wrong. The heavy work runs in fast libraries underneath.
- Python is the most common choice, not the only one. TypeScript is also used for LLM web apps.

## Remember
- Python is **easy to read**, so ideas are quick to try.
- It has the **biggest set of AI libraries**.
- Python is the **glue**; fast C/CUDA code does the heavy maths.

## Related Questions
- [What are modules and packages?](07-What-are-modules-and-packages.md)
- [What is NumPy?](10-What-is-NumPy.md)
- [How do you consume a REST API using Python?](14-How-do-you-consume-a-REST-API-using-Python.md)
- [BONUS: What are async and await?](16-BONUS-What-are-async-and-await-and-why-do-they-matter-for-LLM-apps.md)
