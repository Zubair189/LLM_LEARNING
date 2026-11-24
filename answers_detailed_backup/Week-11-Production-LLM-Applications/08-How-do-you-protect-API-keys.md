# How do you protect API keys?

> Week 11 – Production LLM Applications · Question 8

## Short Answer
Keep API keys **server-side only**. Load them from environment variables or a secrets manager, never hard-code them or commit them to Git, and never ship them to browsers or mobile apps. Give each environment and service its own least-privilege key with spending limits, rotate keys regularly, and scan code and logs for leaks so a leaked key can be revoked quickly.

## Detailed Explanation
**Where keys should and shouldn't live**
| Location | OK? |
|---|---|
| Source code / notebooks | No |
| Git history (even if later deleted) | No; still recoverable |
| Front-end JS / mobile app bundle | No; anyone can extract it |
| Logs, error messages, prompts | No |
| `.env` file (git-ignored) for local dev | Yes, for local only |
| Environment variables in deployment | Yes |
| Secrets manager (AWS Secrets Manager, GCP Secret Manager, Vault, Azure Key Vault) | Yes, best for production |

**Architecture**
```
Browser ──(user session/JWT)──► Your backend ──(ANTHROPIC_API_KEY)──► LLM provider
                                     ▲
                       Secrets manager / env var (injected at deploy)
```
The user authenticates to *your* backend; only the backend holds the provider key.

**Operational practices**
- Separate keys per environment (dev/staging/prod) and per service.
- Set usage and spend limits in the provider console; alert on spikes.
- Rotate on a schedule and immediately after suspected exposure.
- Use pre-commit secret scanners (e.g. `gitleaks`, `detect-secrets`) and GitHub secret scanning.
- Redact secrets in logs and error traces.
- Restrict who can read secrets (IAM) and audit access.

## Example
**Analogy:** Your house key stays in your pocket. You don't tape it to the front door. You give a guest temporary access through a doorman (your backend), not a copy of the key.

```python
# .env  (listed in .gitignore)  ->  ANTHROPIC_API_KEY=sk-ant-...
import os, logging
import anthropic
from dotenv import load_dotenv   # pip install python-dotenv (local dev only)

load_dotenv()
if not os.environ.get("ANTHROPIC_API_KEY"):
    raise RuntimeError("ANTHROPIC_API_KEY is not set")

client = anthropic.Anthropic()   # reads the key from the environment

def redact(s: str) -> str:
    key = os.environ["ANTHROPIC_API_KEY"]
    return s.replace(key, "***REDACTED***")

logging.basicConfig(level=logging.INFO)
logging.info(redact(f"Using key {os.environ['ANTHROPIC_API_KEY']}"))
# -> INFO:root:Using key ***REDACTED***
```

## Common Mistakes / Misconceptions
- "It's a private repo, so committing the key is fine." Repos get forked, cloned and leaked.
- Calling the LLM API directly from a React or mobile app "to save a backend".
- Deleting a leaked key from code without revoking it. The key must be rotated.
- One shared key for everything, so you can't tell who caused a spike or revoke narrowly.

## Key Takeaways
- Keys belong in env vars or a secrets manager, only on the server.
- Use per-environment, least-privilege keys with spend limits.
- Rotate regularly and revoke immediately on leaks.
- Automate leak detection (pre-commit hooks, repo scanning, log redaction).

## Related Questions
- [What is an API key?](../Week-06-Working-with-LLMs-and-Model-APIs/08-What-is-an-API-key.md)
- [What is the role of an API gateway?](02-What-is-the-role-of-an-API-gateway.md)
- [What is rate limiting?](09-What-is-rate-limiting.md)
- [How do you secure an LLM application?](../Week-12-Final-Project-and-Revision/10-How-do-you-secure-an-LLM-application.md)
