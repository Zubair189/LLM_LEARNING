# What is an API key?

> Week 06 – Working with LLMs and Model APIs · Question 8

> Note: The API key is the **secret** that proves who you are, and the inference API ([Q7](07-What-is-an-inference-API.md)) is the **service** it unlocks.

## In One Line
An API key is a **secret code that proves who you are** when your program uses an API, a bit like a password.

## Simple Explanation
When your code calls a paid API, the provider needs to know **who** is calling. The API key tells them.

With every request, the key is sent in a **header** (an extra label on the request). Anthropic uses a header called `x-api-key`.

The provider then:

1. **Checks the key.** Wrong or missing → error **401 Unauthorized**.
2. **Checks your limits.** Too many requests → error **429**.
3. **Runs the request** and adds the cost to **your bill**.

So if someone steals your key, they can spend your money.

How to keep it safe:
- Store it in an **environment variable** (a setting outside your code), like `ANTHROPIC_API_KEY`, or in a `.env` file.
- Add `.env` to `.gitignore` so it never goes to GitHub.
- Never put it in website or mobile app code, where anyone can read it.
- Set a **spending limit** in the provider's console.
- If it leaks, **revoke it** (switch it off) right away and make a new one.

**Link to LLMs:** `anthropic.Anthropic()` automatically reads `ANTHROPIC_API_KEY` from your environment, so the key never appears in your code.

## Real-Life Example
An API key is like a hotel key card. It opens only the doors you are allowed into, every use is charged to your room, and if you lose it, the front desk switches it off and gives you a new one.

## Code Example
```python
import os

key = os.environ.get("ANTHROPIC_API_KEY")     # read the key from the environment
if not key:                                   # stop early if it's missing
    raise SystemExit("Please set ANTHROPIC_API_KEY")

print(f"Using key ...{key[-4:]}")             # show only the last 4 characters, never the whole key

# Output when the key is not set:
# Please set ANTHROPIC_API_KEY
# Output when it is set (e.g.):
# Using key ...ab12
```

Set the key in your terminal first, for example `export ANTHROPIC_API_KEY=...` (macOS/Linux).

**What happened:** the code read the key from outside the program, stopped with a clear message when it was missing, and printed only the last 4 characters so logs never show the full secret.

## Watch Out
- Deleting a leaked key from your code is **not enough**. Old git history still has it. Revoke the key.
- Don't share one key with the whole team. Separate keys make it easy to track and switch off one without breaking others.

## Remember
- API key = **secret that identifies you** and your bill.
- Keep it in **environment variables**, never in code.
- If it leaks, **revoke it** immediately.

## Related Questions
- [What is an inference API?](07-What-is-an-inference-API.md)
- [How do you handle API errors?](../Week-05-Python-for-LLM-Engineering/15-How-do-you-handle-API-errors.md)
- [What are virtual environments and how do you manage dependencies?](../Week-05-Python-for-LLM-Engineering/17-BONUS-What-are-virtual-environments-and-how-do-you-manage-dependencies.md)
- [How do you manage API costs and rate limits?](17-How-do-you-manage-API-costs-and-rate-limits.md)
