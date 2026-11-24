# Why should AI-generated SQL be restricted?

> Week 11 – Production LLM Applications · Question 17

## In One Line
SQL written by an AI is **untrusted code**: it might delete data, leak private data or be plain wrong, so you must limit what it is allowed to do.

## Simple Explanation
**SQL** is the language used to read and change data in a database. Some apps let users ask questions in plain English, and the LLM writes the SQL.

That's handy, but risky. The model might:
- **Destroy data:** `DROP TABLE orders` or `DELETE FROM users`.
- **Leak data:** `SELECT * FROM salaries`, or another customer's rows.
- **Be tricked:** a prompt injection like "...and also run `UPDATE accounts SET balance = 1000000`".
- **Overload the database:** a giant query that slows things down for everyone.
- **Be quietly wrong:** a plausible query with a wrong join that returns wrong numbers.

**How to restrict it, in layers:**
1. **Read-only database user.** The connection the AI uses can only *read*, and only approved tables. This is the most important layer: even if every other check fails, it **cannot** change data.
2. **Check the query before running it:** exactly one statement, must be `SELECT`, only allowed tables. (Real apps use a proper SQL parser like `sqlglot`.)
3. **Add limits:** a maximum number of rows (`LIMIT 100`) and a time limit.
4. **Log every query** with who asked for it.

```
Question → LLM writes SQL → Check (SELECT only, allowed tables) → Run as read-only user → Results
```

## Real-Life Example
You let an intern pull reports from the archive. You give them a visitor badge that only opens the reading room, not the shredder room. A librarian also checks each request slip before fetching files.

## Code Example
A simple keyword-based check. It's good for learning; real apps parse the SQL properly *and* use a read-only user.

```python
import re

ALLOWED_TABLES = {"orders", "products"}
BANNED = ["insert", "update", "delete", "drop", "alter", "create", "grant"]

def is_safe(sql):
    s = sql.strip().rstrip(";").lower()
    if ";" in s:                                   # more than one statement?
        return False, "only one statement allowed"
    if not s.startswith("select"):                 # must be a read-only SELECT
        return False, "only SELECT allowed"
    if any(re.search(rf"\b{w}\b", s) for w in BANNED):
        return False, "banned keyword"
    tables = set(re.findall(r"(?:from|join)\s+(\w+)", s))   # tables it reads
    if not tables <= ALLOWED_TABLES:
        return False, f"table not allowed: {tables - ALLOWED_TABLES}"
    return True, "ok"

for q in ["SELECT id, total FROM orders", "DROP TABLE orders",
          "SELECT 1; DELETE FROM orders", "SELECT * FROM salaries"]:
    print(is_safe(q), "|", q)
# (True, 'ok') | SELECT id, total FROM orders
# (False, 'only SELECT allowed') | DROP TABLE orders
# (False, 'only one statement allowed') | SELECT 1; DELETE FROM orders
# (False, "table not allowed: {'salaries'}") | SELECT * FROM salaries
```

**What happened:** only the safe query on an allowed table passed. The delete, the sneaky second statement and the private table were all blocked.

## Watch Out
- "The system prompt says only write SELECT" is **not** security. Prompts can be bypassed; checks in code and database permissions can't.
- Never let the AI use an admin or owner database account.

## Remember
- Treat AI-written SQL like code from a stranger.
- A **read-only database user** is the safety net.
- Check the query, add row and time limits, and log everything.

## Related Questions
- [What is prompt injection?](06-What-is-prompt-injection.md)
- [How do you implement human approval for sensitive operations?](18-How-do-you-implement-human-approval-for-sensitive-operations.md)
- [How do you prevent an agent from performing unauthorized operations?](../Week-10-AI-Agents-and-Tool-Calling/14-How-do-you-prevent-an-agent-from-performing-unauthorized-operations.md)
