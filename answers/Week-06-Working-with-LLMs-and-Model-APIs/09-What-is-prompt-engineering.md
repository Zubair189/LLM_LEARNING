# What is prompt engineering?

> Week 06 – Working with LLMs and Model APIs · Question 9

## In One Line
Prompt engineering is **writing and testing the instructions you give an LLM** so it gives the answer you want, reliably.

## Simple Explanation
A **prompt** is the text you send to an LLM. The model can only work with what you give it, so a clear prompt gets a better answer.

A good prompt usually has:

1. **Role:** who the model should act as. "You are a support agent for an online shop."
2. **Task:** exactly what to do. "Put the message into one category."
3. **Data, clearly marked:** wrap the input in tags like `<message>...</message>` so the model knows what is instruction and what is data.
4. **Rules:** "Only use: billing, shipping, account, other."
5. **Output format:** "Reply with only the category name."
6. **Examples** (optional): a few input → output pairs. This is called **few-shot prompting**.

Two more useful tricks:
- **Ask it to think step by step** for harder problems.
- **Split big tasks** into several smaller prompts.

**Test it.** Keep a small set of test inputs with the right answers. After every prompt change, run them all to check you didn't break something.

**Link to LLMs:** prompt engineering is usually the **first and cheapest** way to improve results, before trying fine-tuning.

## Real-Life Example
It's like briefing a smart new contractor. "Fix the house" gets you something random. "Paint the kitchen white, by Friday, send me a photo" gets you exactly what you want.

## Code Example
```python
import anthropic

client = anthropic.Anthropic()                       # reads ANTHROPIC_API_KEY
vague = "What category is this: 'I was charged twice for my order'"
clear = """Classify the customer message into exactly one category.
Categories: billing, shipping, account, other.
<message>I was charged twice for my order</message>
Reply with only the category name in lowercase."""

for prompt in (vague, clear):                         # try both prompts
    r = client.messages.create(model="claude-sonnet-5-5", max_tokens=1024,
                               messages=[{"role": "user", "content": prompt}])
    print(repr(r.content[0].text))                    # repr shows the exact text

# Output (e.g.):
# 'This sounds like a billing or payment issue, because the customer ...'
# 'billing'
```

**What happened:** the vague prompt got a chatty sentence. The clear prompt got one clean word that code can use directly.

## Watch Out
- It's not about "magic words". It is mostly **clear instructions + testing**.
- Changing a prompt without a test set often fixes one case and quietly breaks others.

## Remember
- Be **clear and specific**: role, task, data, rules, format.
- **Mark user data** with tags.
- **Test** every change on a few examples.

## Related Questions
- [What is a system prompt?](10-What-is-a-system-prompt.md)
- [What is a user prompt?](11-What-is-a-user-prompt.md)
- [What is few-shot prompting?](18-BONUS-What-is-few-shot-prompting.md)
- [What is structured output?](15-What-is-structured-output.md)
