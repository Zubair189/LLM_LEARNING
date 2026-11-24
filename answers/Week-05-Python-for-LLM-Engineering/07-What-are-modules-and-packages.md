# What are modules and packages?

> Week 05 – Python for LLM Engineering · Question 7

## In One Line
A **module** is one Python file you can import, and a **package** is a folder of modules (or a library you install with `pip`).

## Simple Explanation
As a project grows, putting all your code in one file gets messy. So Python lets you split it up.

1. **Module:** a single `.py` file, like `prompts.py`. It holds functions, classes and variables.
2. **Package:** a folder that groups several modules, like `rag/` containing `retriever.py`.
3. **Installed package:** a library other people wrote, downloaded from **PyPI** (the online store of Python libraries) with `pip install anthropic`.

You use any of them with the **`import`** keyword:

- `import json` brings in a whole module.
- `from prompts import user_message` brings in one name.
- `import numpy as np` gives it a short nickname.

Python also ships with a free set of modules called the **standard library**, for example `json`, `pathlib` and `asyncio`. No install needed.

**Link to LLMs:** `import anthropic` is how you bring the Claude SDK into your code after `pip install anthropic`.

## Real-Life Example
Modules are chapters, a package is a book that groups the chapters, and PyPI is a public library where you borrow other people's books.

## Code Example
Two files in the same folder.

```python
# file: prompts.py  (our own module)
SYSTEM_PROMPT = "You are a helpful assistant."      # a variable in the module

def user_message(text):                              # a function in the module
    return {"role": "user", "content": text}
```

```python
# file: main.py
import json                                          # a module from the standard library
from prompts import SYSTEM_PROMPT, user_message      # our own module (prompts.py)

msg = user_message("Hi")                             # use the imported function
print(SYSTEM_PROMPT)
print(json.dumps(msg))                               # use the json module

# Output (python3 main.py):
# You are a helpful assistant.
# {"role": "user", "content": "Hi"}
```

**What happened:** `main.py` reused code from `prompts.py` and from Python's built-in `json` module, without copying anything.

## Watch Out
- Don't name your own file after a library (`json.py`, `anthropic.py`). Python imports your file instead of the real one, and things break.
- `ModuleNotFoundError` after installing usually means you installed into a different Python. Activate your virtual environment and use `python -m pip install ...`.

## Remember
- **Module** = one `.py` file.
- **Package** = folder of modules, or a library from PyPI.
- Use **`import`** to reuse code instead of copying it.

## Related Questions
- [Why is Python widely used in AI development?](01-Why-is-Python-widely-used-in-AI-development.md)
- [What is object-oriented programming?](06-What-is-object-oriented-programming.md)
- [BONUS: What are virtual environments and how do you manage dependencies?](17-BONUS-What-are-virtual-environments-and-how-do-you-manage-dependencies.md)
