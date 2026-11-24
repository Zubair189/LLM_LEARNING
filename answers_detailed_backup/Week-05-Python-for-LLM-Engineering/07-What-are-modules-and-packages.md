# What are modules and packages?

> Week 05 – Python for LLM Engineering · Question 7

## Short Answer
A **module** is a single `.py` file containing code (functions, classes, variables) that you can `import`. A **package** is a directory of modules, organised into a namespace (usually marked with an `__init__.py`), such as `myapp.llm.client`. The word "package" also refers to a **distribution** you install from PyPI with `pip` or `uv`, for example `pip install anthropic`. Modules and packages let you split a project into reusable, testable pieces and pull in other people's libraries.

## Detailed Explanation

```
llm_app/                  <- project root
├── pyproject.toml        <- dependencies and metadata
└── src/llm_app/          <- package
    ├── __init__.py
    ├── prompts.py        <- module
    ├── clients.py        <- module
    └── rag/              <- sub-package
        ├── __init__.py
        └── retriever.py
```

| Term | What it is | Example |
|---|---|---|
| Module | One `.py` file | `prompts.py`, or the standard library's `json` |
| Package (import) | Folder of modules | `llm_app.rag` |
| Distribution package | Installable bundle on PyPI | `anthropic`, `numpy`, `transformers` |
| Standard library | Ships with Python | `json`, `asyncio`, `pathlib`, `logging` |

**Import styles:**
```python
import json                         # whole module
from pathlib import Path            # one name
import numpy as np                  # alias (a convention)
from llm_app.rag.retriever import search   # absolute import (preferred)
from .prompts import SYSTEM_PROMPT  # relative import inside a package
```

**How Python finds modules:** it searches `sys.path`, which includes the script's directory, the standard library and the active environment's `site-packages`. That is why the **virtual environment** you activate decides which packages are available.

**`if __name__ == "__main__":`** runs code only when a file is executed directly, not when it's imported.

**Install name ≠ import name sometimes:** `pip install python-dotenv` → `import dotenv`, and `pip install scikit-learn` → `import sklearn`.

## Example
**Analogy:** modules are chapters, a package is a book that groups the chapters, and PyPI is the library you borrow other people's books from.

```python
# file: prompts.py  (a module)
SYSTEM_PROMPT = "You are a concise assistant."

def user_message(text: str) -> dict:
    return {"role": "user", "content": text}

if __name__ == "__main__":
    print(user_message("test"))   # runs only with: python prompts.py
```

```python
# file: main.py  (imports the module above, plus an installed package)
import anthropic                       # installed: pip install anthropic
from prompts import SYSTEM_PROMPT, user_message

client = anthropic.Anthropic()
resp = client.messages.create(
    model="claude-haiku-5-5",
    max_tokens=100,
    system=SYSTEM_PROMPT,
    messages=[user_message("Name one Python package for data analysis.")],
)
print(resp.content[0].text)   # e.g. "Pandas."
```

## Common Mistakes / Misconceptions
- **Naming your file after a library** (`json.py`, `anthropic.py`, `numpy.py`). Your file shadows the real package and imports break in confusing ways.
- **`ModuleNotFoundError` after `pip install`**, because the package went into a different interpreter or environment. Use `python -m pip install ...` and activate your virtual environment first.
- **Circular imports** (a imports b, b imports a). Move shared code into a third module.
- **`from module import *`** pollutes the namespace and hides where names come from.

## Key Takeaways
- Module = file. Package = folder of modules. Distribution = something installable from PyPI.
- Python finds imports via `sys.path`, which your active environment controls.
- Use absolute imports, clear names, and the `__main__` guard.
- Pin dependencies in `pyproject.toml` / a lock file for reproducible LLM apps.

## Related Questions
- [Why is Python widely used in AI development?](01-Why-is-Python-widely-used-in-AI-development.md)
- [What is object-oriented programming?](06-What-is-object-oriented-programming.md)
- [BONUS: What are virtual environments and how do you manage dependencies?](17-BONUS-What-are-virtual-environments-and-how-do-you-manage-dependencies.md)
