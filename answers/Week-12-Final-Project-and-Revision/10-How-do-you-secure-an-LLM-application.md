# How do you secure an LLM application?

> Week 12 – Final Project and Revision · Question 10

## In One Line
You secure an LLM app with **many layers of protection**, assuming the model *can* be tricked and making sure that a trick **can't cause serious harm**.

## Simple Explanation
The model itself is **not** a security wall. Attackers can talk it into things (prompt injection), and it can make mistakes. So the real security lives in **your code and permissions** around it. This approach is called **defence in depth**: many layers, so if one fails, another catches the problem.

```
User → Gateway (login, limits) → Input checks → Documents (only allowed ones) → LLM → Tool checks (allow-list, read-only, human OK) → Output checks → User
                                                                                                      └─ logs & alerts at every step
```

**Step by step, layer by layer:**

1. **Front door (gateway).** Check who the user is, apply **rate limits** and size limits, and use HTTPS.
2. **Secrets.** API keys and database passwords live in environment variables or a secrets manager on the server. They are **never** in the browser, Git, or the prompt.
3. **Input checks.** Limit length, hide private data (like card numbers), and look for obvious injection phrases.
4. **Data access.** In RAG, only search documents **this user is allowed to see**. Wrap retrieved text in tags and tell the model it is *data*, not instructions, because documents can hide tricks (indirect injection).
5. **Model instructions.** Write a clear system prompt. This helps, but don't rely on it alone.
6. **Actions (the most important layer).**
   - Give the model only the tools it truly needs (**least privilege**).
   - Check every tool input.
   - Use a **read-only** database user, and check any AI-written SQL.
   - Require **human approval** for payments, deletes or emails.
7. **Output checks.** Check the format, look for leaked private data or secrets, and never run model output as code or show it as raw HTML.
8. **Watch.** Log every request and tool call, alert on unusual patterns, and regularly try to attack your own app (**red-teaming**).

**Main threats and their fixes:**
- **Direct injection** ("ignore your rules") → input checks, limited tools.
- **Indirect injection** (tricks hidden in documents) → treat documents as data, limit tools, require approval.
- **Data leaks** (seeing another user's files) → permission filters before retrieval, output scanning.
- **Too much power** (an agent deletes records) → least privilege, read-only roles, human approval.
- **Leaked keys** → keep keys on the server, rotate them.
- **Huge bills from abuse** → rate limits, token limits, per-user budgets.

## Real-Life Example
A bank. Guards at the door check ID (login, rate limits). Tellers follow rules but can be fooled by a clever customer (the LLM). The vault needs two keys no matter what the teller says (least privilege and human approval). Cameras record everything (logs).

## Watch Out
- "A strong system prompt is enough" is wrong. Prompts can be bypassed; code and permissions can't be talked out of things.
- Internal documents aren't automatically safe. Anyone who can edit a document can hide instructions in it.

## Remember
- **The model is not a security boundary**; your code is.
- Least privilege + human approval **limit the damage** of any successful trick.
- Treat user input, documents and model output as **untrusted**, and log everything.

## Related Questions
- [What is prompt injection?](../Week-11-Production-LLM-Applications/06-What-is-prompt-injection.md)
- [What is an indirect prompt injection attack?](../Week-11-Production-LLM-Applications/07-What-is-an-indirect-prompt-injection-attack.md)
- [How do you protect API keys?](../Week-11-Production-LLM-Applications/08-How-do-you-protect-API-keys.md)
- [Why should AI-generated SQL be restricted?](../Week-11-Production-LLM-Applications/17-Why-should-AI-generated-SQL-be-restricted.md)
- [What are guardrails in LLM applications?](../Week-11-Production-LLM-Applications/19-BONUS-What-are-guardrails-in-LLM-applications.md)
- [How do you prevent an agent from performing unauthorized operations?](../Week-10-AI-Agents-and-Tool-Calling/14-How-do-you-prevent-an-agent-from-performing-unauthorized-operations.md)
