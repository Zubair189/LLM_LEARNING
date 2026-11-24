# How do you protect API keys?

> Week 11 – Production LLM Applications · Question 8

## In One Line
Keep API keys **secret and on the server only**: load them from environment variables or a secrets manager, and never put them in code, Git, or the browser.

## Simple Explanation
An **API key** is like a password that lets your app use the LLM provider. Anyone who has it can spend **your** money. So it must stay hidden.

**Where a key should NOT be:**
- In your source code or notebooks.
- In Git, even if you delete it later (old versions are still saved).
- In front-end JavaScript or a mobile app (anyone can dig it out).
- In logs, error messages or prompts.

**Where a key SHOULD be:**
- In an **environment variable**: a setting outside your code that the program reads when it starts, e.g. `ANTHROPIC_API_KEY`.
- In a `.env` file for local testing, as long as that file is listed in `.gitignore`.
- In a **secrets manager** (such as AWS Secrets Manager or Vault) for real production apps.

**How it fits together:**

```
Browser → (user login) → Your backend → (API key) → LLM provider
```

The user logs in to *your* server. Only your server holds the provider key.

**Good habits:**
- Use **different keys** for testing and production.
- Set **spending limits** and alerts in the provider's dashboard.
- **Rotate** keys (replace them) regularly, and right away if one leaks.
- Use tools like `gitleaks` that warn you before you commit a secret.

## Real-Life Example
Your house key stays in your pocket. You don't tape it to the front door. When a guest visits, a doorman (your backend) lets them in; they never get a copy of the key.

## Code Example
Read the key from the environment and never print it in full.

```python
import os

key = os.environ.get("ANTHROPIC_API_KEY")       # read the key from the environment
if key is None:
    raise SystemExit("ANTHROPIC_API_KEY is not set")   # fail early with a clear message

def mask(secret):
    return "****" + secret[-4:]                  # show only the last 4 characters

print("Key loaded:", mask(key))                  # safe to print in a log
# client = anthropic.Anthropic()                 # the SDK reads the same variable itself

# With ANTHROPIC_API_KEY=demo-placeholder-1234 set:  Key loaded: ****1234
# With no key set:                                   ANTHROPIC_API_KEY is not set
```

**What happened:** the key never appears in the code itself, and the log shows only its last 4 characters.

## Watch Out
- "It's a private repo, so it's fine" is wrong. Repos get copied, shared and leaked.
- Deleting a leaked key from your code is not enough. You must **revoke** it in the provider dashboard and make a new one.

## Remember
- Keys live in **environment variables or a secrets manager**, only on the server.
- Use separate keys with spending limits.
- If a key leaks, **revoke and replace** it immediately.

## Related Questions
- [What is an API key?](../Week-06-Working-with-LLMs-and-Model-APIs/08-What-is-an-API-key.md)
- [What is the role of an API gateway?](02-What-is-the-role-of-an-API-gateway.md)
- [What is rate limiting?](09-What-is-rate-limiting.md)
- [How do you secure an LLM application?](../Week-12-Final-Project-and-Revision/10-How-do-you-secure-an-LLM-application.md)
