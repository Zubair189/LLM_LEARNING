# Why should AI-generated SQL be restricted?

> Week 11 – Production LLM Applications · Question 17

## Short Answer
LLM-generated SQL is untrusted code. A model can misunderstand the request, hallucinate columns, or be steered by prompt injection into `DROP`, `DELETE` or `UPDATE` statements, queries that leak other users' data, or expensive full-table scans. Restrict it in layers: connect with a **read-only database role** limited to approved views, parse and validate each query (e.g. with `sqlglot`) so that only a single `SELECT` on allowlisted tables gets through, add row limits and timeouts, and log every query.

## Detailed Explanation
**What can go wrong**
| Risk | Example |
|---|---|
| Destructive statements | `DROP TABLE orders;` or `DELETE FROM users` |
| Data leakage | `SELECT * FROM salaries` or another tenant's rows |
| Injection via user/text | "...and also run `UPDATE accounts SET balance=1e9`" |
| Stacked queries | `SELECT 1; DROP TABLE x;` |
| Resource abuse | Cartesian joins or unbounded scans that slow the DB for everyone |
| Silent wrong answers | Plausible SQL with a wrong join, giving wrong numbers |

**Defence in depth**
```
NL question ─► LLM generates SQL (given only allowed schema)
                   │
                   ▼
        [1] Parse with sqlglot: exactly 1 statement, SELECT only
        [2] Table/column allowlist; no forbidden functions
        [3] Enforce LIMIT; tenant filter / row-level security
                   │
                   ▼
        [4] Execute as READ-ONLY role, on replica/views, statement_timeout
                   │
                   ▼
        [5] Log SQL + user + row count; human review for anything else
```
Layer **[4]** is the safety net. Even if the validator has a bug, the DB role physically cannot write. Example Postgres setup:
```sql
CREATE ROLE llm_reader LOGIN PASSWORD '...';
GRANT SELECT ON reporting.orders_v, reporting.customers_v TO llm_reader;
ALTER ROLE llm_reader SET default_transaction_read_only = on;
ALTER ROLE llm_reader SET statement_timeout = '5s';
```

## Example
**Analogy:** You let an intern pull reports from the archive, but you give them a visitor badge that only opens the reading room, not the shredder room. A librarian also checks each request slip.

```python
import sqlglot
from sqlglot import exp

ALLOWED_TABLES = {"orders_v", "customers_v"}
FORBIDDEN = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command)

def validate_sql(sql: str, max_rows: int = 100) -> str:
    statements = sqlglot.parse(sql, read="postgres")
    if len(statements) != 1 or statements[0] is None:
        raise ValueError("exactly one statement allowed")
    stmt = statements[0]
    if not isinstance(stmt, exp.Select) or any(stmt.find_all(*FORBIDDEN)):
        raise ValueError("only SELECT queries allowed")
    tables = {t.name for t in stmt.find_all(exp.Table)}
    if not tables <= ALLOWED_TABLES:
        raise ValueError(f"table not allowed: {tables - ALLOWED_TABLES}")
    if stmt.args.get("limit") is None:
        stmt = stmt.limit(max_rows)
    return stmt.sql(dialect="postgres")

print(validate_sql("SELECT name FROM customers_v"))
# -> SELECT name FROM customers_v LIMIT 100
for bad in ["DROP TABLE orders_v", "SELECT 1; DELETE FROM orders_v", "SELECT * FROM salaries"]:
    try:
        validate_sql(bad)
    except ValueError as e:
        print("blocked:", e)
# -> blocked: only SELECT queries allowed
# -> blocked: exactly one statement allowed
# -> blocked: table not allowed: {'salaries'}
```
Then run the validated SQL with a connection that uses the `llm_reader` credentials from an environment variable.

## Common Mistakes / Misconceptions
- "The system prompt says only write SELECT queries, so we're safe." Prompts are not security controls.
- Regex checks like `if "DROP" in sql`. They are easy to bypass and also block legitimate text; parse the SQL instead.
- Using the app's admin/owner DB credentials for LLM queries.
- Forgetting multi-tenant filters, so users can query other customers' data.

## Key Takeaways
- Treat generated SQL as untrusted input, like user-submitted code.
- Read-only role + allowlisted views is the non-negotiable safety net.
- Parse and validate (single SELECT, allowed tables, LIMIT) before execution.
- Add timeouts, logging and human review for anything outside the allowlist.

## Related Questions
- [What is prompt injection?](06-What-is-prompt-injection.md)
- [How do you implement human approval for sensitive operations?](18-How-do-you-implement-human-approval-for-sensitive-operations.md)
- [How do you prevent an agent from performing unauthorized operations?](../Week-10-AI-Agents-and-Tool-Calling/14-How-do-you-prevent-an-agent-from-performing-unauthorized-operations.md)
- [What is tool calling?](../Week-10-AI-Agents-and-Tool-Calling/03-What-is-tool-calling.md)
