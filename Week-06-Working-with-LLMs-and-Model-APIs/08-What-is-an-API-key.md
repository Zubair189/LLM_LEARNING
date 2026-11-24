# What is an API key?

> Week 06 – Working with LLMs and Model APIs · Question 8

> Note: The API key is the **credential**; the inference API ([Q7](07-What-is-an-inference-API.md)) is the **service** it unlocks.

## Short Answer
An API key is a secret string that identifies your account or project when you call an API, so the provider can authenticate you, enforce rate limits and bill your usage. It is sent with each request, typically in an HTTP header (Anthropic uses `x-api-key`; many others use `Authorization: Bearer ...`). Treat it like a password: keep it in environment variables or a secrets manager, never in code or git.

## Detailed Explanation
What a key does on each request:

```
 request + key ──► provider checks key ──► valid? ─ no ─► 401 Unauthorized
                                             │
                                            yes ─► check quota/rate limit ─► 429 if exceeded
                                             │
                                             └──► run model, bill usage to that key's workspace
```

Good practice:

| Practice | Why |
|---|---|
| Store in env var (`ANTHROPIC_API_KEY`) or secrets manager | Keeps keys out of source code |
| Add `.env` to `.gitignore` | Prevents accidental commits |
| One key per app/environment | Revoke one without breaking others; track spend |
| Set spend limits in the console | Caps damage if leaked |
| Rotate periodically and immediately on leak | Limits exposure window |
| Never ship keys in frontend/mobile code | Anyone can extract them; call the LLM from your backend |

If a key leaks (e.g. pushed to a public repo), automated scanners may find it within minutes. Revoke it in the provider console first, then create a new one.

## Example
**Analogy:** An API key is like a hotel key card — it opens the doors you are allowed into, every use is logged to your room's bill, and if lost you deactivate it at the front desk.

```python
# .env file (never commit):  ANTHROPIC_API_KEY=your-key-here
# pip install anthropic python-dotenv
import os
from dotenv import load_dotenv
import anthropic

load_dotenv()  # copies values from .env into os.environ

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise SystemExit("Set ANTHROPIC_API_KEY in your environment or .env file")

client = anthropic.Anthropic()  # automatically reads ANTHROPIC_API_KEY
try:
    resp = client.messages.create(
        model="claude-haiku-5-5", max_tokens=50,
        messages=[{"role": "user", "content": "Say OK"}])
    print(resp.content[0].text)          # Expected: OK
except anthropic.AuthenticationError:
    print("Invalid or revoked API key")  # raised on HTTP 401
```

Masking a key in logs:
```python
key = os.environ["ANTHROPIC_API_KEY"]
print(f"Using key ...{key[-4:]}")  # Expected: Using key ...ab12 (last 4 chars only)
```

## Common Mistakes / Misconceptions
- **Hardcoding keys in notebooks or scripts** that later get shared or committed.
- **Putting keys in browser JavaScript.** Users can read them in dev tools.
- **Thinking deleting the commit removes the leak.** Git history and forks keep it; revoke the key.
- **Sharing one key across the whole team.** You lose per-user/app tracking and must rotate for everyone.

## Key Takeaways
- An API key authenticates you and ties usage to your bill and limits.
- Load it from environment variables or a secrets manager.
- Never commit or expose it client-side.
- Use separate keys, spend limits and rotation.

## Related Questions
- [What is an inference API?](07-What-is-an-inference-API.md)
- [How do you handle API errors?](../Week-05-Python-for-LLM-Engineering/15-How-do-you-handle-API-errors.md)
- [What are virtual environments and how do you manage dependencies?](../Week-05-Python-for-LLM-Engineering/17-BONUS-What-are-virtual-environments-and-how-do-you-manage-dependencies.md)
- [How do you manage API costs and rate limits?](17-How-do-you-manage-API-costs-and-rate-limits.md)
