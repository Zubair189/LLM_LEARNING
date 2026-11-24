# What are virtual environments and how do you manage dependencies?

> Week 05 – Python for LLM Engineering · Question 17 · **BONUS**
>
> Bonus question (added)

## Short Answer
A **virtual environment** is an isolated folder containing its own Python interpreter link and its own `site-packages`, so each project can have its own package versions without conflicting with other projects or the system Python. You **manage dependencies** by declaring them (in `pyproject.toml`, or `requirements.txt`), installing them into the environment, and **pinning exact versions in a lock file** so everyone (teammates, CI, production) gets the same setup. Common tools are the built-in `venv` + `pip`, and **`uv`**, a fast all-in-one tool that has become very popular.

## Detailed Explanation

**Why isolation matters:**
```
Project A (old RAG demo)  ──► langchain 0.1, numpy 1.26
Project B (new agent)     ──► langchain 1.x, numpy 2.x
Global install?  ✗ conflict      Separate venvs?  ✓ both work
```

**Tool landscape:**

| Tool | What it does |
|---|---|
| `venv` + `pip` | Built in. `python -m venv .venv`, then `pip install`. Use `pip freeze > requirements.txt` to pin. |
| **`uv`** | Very fast Rust-based tool: creates venvs, resolves and installs, writes a lock file (`uv.lock`), manages Python versions. |
| Poetry / PDM / Hatch | Project managers built around `pyproject.toml` with lock files. |
| Conda / mamba | Also manage non-Python libraries (CUDA, MKL). Common in data science and GPU setups. |
| Docker | Isolates the whole OS layer for deployment. |

**Key files:**
- `pyproject.toml`: the modern standard for declaring a project and its dependency *ranges* (`anthropic>=0.70`).
- Lock file (`uv.lock`, `poetry.lock`, or a pinned `requirements.txt`): *exact* versions of every package, including transitive dependencies.
- `.env`: secrets such as `ANTHROPIC_API_KEY`. **Not** committed (add it to `.gitignore`), and not the same thing as `.venv`.

**Dependency hygiene for LLM projects:** LLM SDKs and frameworks release often and sometimes include breaking changes, so pin versions, update deliberately and re-run your evals after upgrading. GPU libraries (`torch`) need builds that match your CUDA version.

## Example
**Analogy:** each virtual environment is a separate toolbox for each job. The plumbing toolbox doesn't get mixed up with the electrical one, and the lock file is the packing list that lets anyone build an identical toolbox.

```bash
# Option A: built-in tools
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install anthropic pydantic httpx
python -m pip freeze > requirements.txt

# Option B: uv (fast, lock file included)
uv init llm-app && cd llm-app
uv add anthropic pydantic httpx    # updates pyproject.toml + uv.lock, creates .venv
uv run python main.py              # runs inside the project's environment
```

```python
# check_env.py: verify you're in the right environment
import sys
from importlib.metadata import version

print("in venv:", sys.prefix != sys.base_prefix)   # in venv: True
print("python:", sys.executable)                   # e.g. /path/to/llm-app/.venv/bin/python
print("anthropic:", version("anthropic"))          # e.g. anthropic: 0.x.y
```

## Common Mistakes / Misconceptions
- **Installing into the global/system Python** (or with `sudo pip`), which breaks other projects or even OS tools.
- **Committing `.venv/` to git.** Commit `pyproject.toml` and the lock file instead; environments are rebuilt.
- **`pip install` and `python` pointing at different interpreters.** Use `python -m pip` and check that `sys.executable` is what you expect.
- **Unpinned dependencies**, so "it worked yesterday" breaks after a new SDK release.

## Key Takeaways
- One virtual environment per project. Never install project packages globally.
- Declare dependencies in `pyproject.toml` and pin exact versions in a lock file.
- `uv` is a fast, modern default. `venv` + `pip` works everywhere.
- Keep secrets in `.env` (ignored by git), separate from the `.venv` folder.

## Related Questions
- [What are modules and packages?](07-What-are-modules-and-packages.md)
- [Why is Python widely used in AI development?](01-Why-is-Python-widely-used-in-AI-development.md)
- [What is an API key?](../Week-06-Working-with-LLMs-and-Model-APIs/08-What-is-an-API-key.md)
