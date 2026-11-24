# What are virtual environments and how do you manage dependencies?

> Week 05 – Python for LLM Engineering · Question 17 · **BONUS**
>
> Bonus question (added)

## In One Line
A virtual environment is a **private box of Python packages for one project**, so different projects don't fight over versions.

## Simple Explanation
**Dependencies** are the packages your project needs, like `anthropic` or `pandas`.

The problem: an old project might need version 1 of a package, while a new project needs version 2. If you install everything into one shared Python, they clash.

A **virtual environment** (often a folder called `.venv`) solves this. Each project gets its own separate set of packages.

Managing dependencies means three steps:

1. **Create** an environment for the project.
2. **Install** packages into it.
3. **Write them down** so others can rebuild the same setup:
   - `pyproject.toml` lists what you need.
   - A **lock file** (like `uv.lock` or a pinned `requirements.txt`) records the **exact** versions.

Common tools:
- **`venv` + `pip`:** built into Python.
- **`uv`:** a newer, very fast tool that does everything, including the lock file.

**Link to LLMs:** LLM libraries change quickly. Pinning versions stops your app breaking when a new version comes out.

## Real-Life Example
Each virtual environment is a separate toolbox for each job. The plumbing tools never get mixed up with the electrical tools. The lock file is the packing list that lets anyone build an identical toolbox.

## Code Example
```bash
# Option A: built-in tools
python -m venv .venv                     # create the environment
source .venv/bin/activate                # turn it on (Windows: .venv\Scripts\activate)
python -m pip install anthropic pandas   # install into this project only
python -m pip freeze > requirements.txt  # write down exact versions

# Option B: uv
uv init llm-app && cd llm-app            # new project
uv add anthropic pandas                  # install + update pyproject.toml and uv.lock
uv run python main.py                    # run inside the project's environment
```

```python
import sys                                         # info about the running Python

print("in venv:", sys.prefix != sys.base_prefix)   # True inside a virtual environment
print("python:", sys.executable)                   # which Python is running

# Output (inside an activated venv, e.g.):
# in venv: True
# python: /path/to/llm-app/.venv/bin/python
```

**What happened:** we created a project-only box of packages, saved the exact versions, and then checked that Python is really running from inside it.

## Watch Out
- Don't commit the `.venv` folder to git. Commit `pyproject.toml` and the lock file; anyone can rebuild the environment from them.
- `.venv` (packages) is not `.env` (secret keys). Keep `.env` out of git too.

## Remember
- **One virtual environment per project.**
- **Pin exact versions** in a lock file.
- `venv` + `pip` works everywhere; `uv` is fast and modern.

## Related Questions
- [What are modules and packages?](07-What-are-modules-and-packages.md)
- [Why is Python widely used in AI development?](01-Why-is-Python-widely-used-in-AI-development.md)
- [What is an API key?](../Week-06-Working-with-LLMs-and-Model-APIs/08-What-is-an-API-key.md)
