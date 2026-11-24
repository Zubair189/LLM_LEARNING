# What is Pandas?

> Week 05 – Python for LLM Engineering · Question 11

## In One Line
Pandas is a Python library for working with **tables of data**, like a spreadsheet you control with code.

## Simple Explanation
Pandas has two main things:

1. **DataFrame:** a whole table with rows and named columns.
2. **Series:** a single column of that table.

What you usually do with Pandas:

1. **Load** data from a file, like `pd.read_csv("data.csv")`.
2. **Look** at it with `df.head()` (first few rows).
3. **Filter** rows, like "only rows where tokens > 50".
4. **Group** rows and summarise them, like "average score per model".
5. **Save** the result back to a file.

Pandas is built on top of NumPy, so column maths is fast.

**Link to LLMs:** Pandas is the everyday tool for cleaning training data, comparing model test results, and adding up token usage and cost.

## Real-Life Example
Pandas is like Excel, but instead of clicking, you write the steps as code. That means you can repeat the same steps on a new file of a million rows with one command.

## Code Example
```python
import pandas as pd                                  # the usual short name

log = pd.DataFrame({                                 # build a small table
    "model":   ["haiku", "sonnet", "haiku", "sonnet"],
    "correct": [True, True, False, True],
    "tokens":  [40, 55, 80, 90],
})

print(log.head(2))                                   # look at the first 2 rows
print(log[log["tokens"] > 50])                       # filter: rows with > 50 tokens
print(log.groupby("model")["correct"].mean())        # accuracy per model

# Output:
#     model  correct  tokens
# 0   haiku     True      40
# 1  sonnet     True      55
#     model  correct  tokens
# 1  sonnet     True      55
# 2   haiku    False      80
# 3  sonnet     True      90
# model
# haiku     0.5
# sonnet    1.0
# Name: correct, dtype: float64
```

**What happened:** we made a small test log, picked out the long answers, and worked out that "haiku" was right 50% of the time and "sonnet" 100% (True counts as 1, False as 0).

## Watch Out
- Avoid looping over rows one by one. Work on whole columns instead; it is much faster.
- To change values in filtered rows, use `df.loc[condition, "column"] = value`, not `df[condition]["column"] = value`.

## Remember
- **DataFrame** = table, **Series** = column.
- Load → look → filter → group → save.
- Great for **datasets, test results and cost logs**.

## Related Questions
- [What is NumPy?](10-What-is-NumPy.md)
- [What is JSON?](09-What-is-JSON.md)
- [How do you manage API costs and rate limits?](../Week-06-Working-with-LLMs-and-Model-APIs/17-How-do-you-manage-API-costs-and-rate-limits.md)
