# What is Pandas?

> Week 05 – Python for LLM Engineering · Question 11

## Short Answer
Pandas is Python's most widely used library for working with **tabular data**. Its main structures are the **`DataFrame`** (a table with labelled columns, like a spreadsheet or SQL table) and the **`Series`** (a single column). It reads and writes CSV, Excel, JSON/JSONL, Parquet and SQL, and makes filtering, grouping, joining and cleaning data straightforward. In LLM engineering it's the everyday tool for preparing datasets, analysing evaluation results, and tracking token usage and cost logs.

## Detailed Explanation

```
DataFrame
            prompt              label      tokens   <- columns (each one a Series)
index 0   "Great phone"       positive      12
      1   "Broke in a day"    negative      15
```

**Common operations:**

| Task | Code |
|---|---|
| Load | `pd.read_csv("f.csv")`, `pd.read_json("f.jsonl", lines=True)` |
| Inspect | `df.head()`, `df.info()`, `df.describe()` |
| Select | `df["label"]`, `df[["prompt", "label"]]`, `df.loc[row, col]` |
| Filter | `df[df["tokens"] > 100]` |
| New column | `df["cost"] = df["tokens"] * rate` |
| Apply a function | `df["prompt"].apply(len)` (slow for big data; prefer vectorised operations) |
| Group | `df.groupby("model")["latency"].mean()` |
| Missing data | `df.isna().sum()`, `df.dropna()`, `df.fillna(0)` |
| Save | `df.to_csv(...)`, `df.to_parquet(...)`, `df.to_json(..., orient="records", lines=True)` |

**Typical LLM uses:**
- Cleaning and deduplicating prompt/response pairs before fine-tuning.
- Comparing models in an evaluation: accuracy and latency per model or prompt version.
- Analysing usage logs: tokens per user per day, and cost estimates.
- Exporting to JSONL for batch APIs or fine-tuning.

Pandas is built on NumPy. For very large data, alternatives like **Polars** or **DuckDB** are popular, but Pandas remains the common default.

## Example
**Analogy:** Pandas is Excel you control with code. Every step can be repeated, versioned and run over a million rows.

```python
import pandas as pd

# An evaluation log, e.g. collected from resp.usage and a grader
log = pd.DataFrame({
    "model":         ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-sonnet-5-5"],
    "correct":       [True, True, False, True],
    "input_tokens":  [120, 120, 300, 300],
    "output_tokens": [40, 55, 80, 90],
})

summary = log.groupby("model").agg(
    accuracy=("correct", "mean"),
    avg_output_tokens=("output_tokens", "mean"),
)
print(summary)
#                    accuracy  avg_output_tokens
# model
# claude-haiku-5-5        0.5               60.0
# claude-sonnet-5-5       1.0               72.5

# ILLUSTRATIVE price per million output tokens: check the provider's pricing page
ILLUSTRATIVE_PRICE_PER_MTOK = {"claude-haiku-5-5": 5.0, "claude-sonnet-5-5": 15.0}
log["est_output_cost"] = log["output_tokens"] * log["model"].map(ILLUSTRATIVE_PRICE_PER_MTOK) / 1_000_000
print(round(log["est_output_cost"].sum(), 6))   # 0.002775
```

## Common Mistakes / Misconceptions
- **`SettingWithCopyWarning` / chained assignment.** `df[df.x > 1]["y"] = 0` may not change `df`. Use `df.loc[df.x > 1, "y"] = 0`. (Under Copy-on-Write, the default since pandas 3.0, chained assignment never updates the original.)
- **Iterating with `iterrows()`** over big frames. Use vectorised column operations instead.
- **Ignoring dtypes.** Numbers read as `object`/strings break maths. Check `df.dtypes`.
- **Loading huge JSONL into memory at once.** Use `chunksize`, Parquet, or Polars/DuckDB.

## Key Takeaways
- `DataFrame` = table, `Series` = column. Pandas is built on NumPy.
- Load → inspect → clean → transform → aggregate → save is the standard workflow.
- Essential for LLM datasets, evaluation analysis and cost/usage reporting.
- Prefer vectorised operations and `.loc` for correct, fast code.

## Related Questions
- [What is NumPy?](10-What-is-NumPy.md)
- [What is JSON?](09-What-is-JSON.md)
- [How do you manage API costs and rate limits?](../Week-06-Working-with-LLMs-and-Model-APIs/17-How-do-you-manage-API-costs-and-rate-limits.md)
