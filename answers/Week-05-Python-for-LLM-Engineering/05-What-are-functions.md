# What are functions?

> Week 05 – Python for LLM Engineering · Question 5

## In One Line
A function is a **named, reusable block of code**: you give it inputs, it does a job, and it gives back a result.

## Simple Explanation
Instead of writing the same steps again and again, you put them in a function once and call it by name.

The parts of a function:

1. **`def`** starts the function and gives it a name, like `build_prompt`.
2. **Parameters** are the inputs it expects, written in brackets.
3. **Default values** make an input optional, like `tone="friendly"`.
4. **`return`** sends a result back to whoever called the function.

If a function has no `return`, it gives back `None` (nothing).

Why functions help:
- **No repetition:** write once, use many times.
- **Clear names:** `call_llm()` says what the code does.
- **Easy testing:** you can check one small piece at a time.

**Link to LLMs:** in "tool calling", you describe your Python functions to an LLM, and the model asks your code to run them.

## Real-Life Example
A function is like a recipe card. You give it ingredients (inputs), follow the steps, and get a dish (the return value). You can use the same card again whenever you want.

## Code Example
```python
def build_prompt(question, tone="friendly"):     # define: 2 inputs, one has a default
    """Make a prompt for an LLM."""             # docstring: what it does
    return f"Answer in a {tone} tone: {question}"  # send the result back

p1 = build_prompt("What is Python?")             # use the default tone
p2 = build_prompt("What is JSON?", tone="short") # override the default
print(p1)
print(p2)

def no_return():
    x = 1 + 1                                    # does work but returns nothing

print(no_return())                               # functions without return give None

# Output:
# Answer in a friendly tone: What is Python?
# Answer in a short tone: What is JSON?
# None
```

**What happened:** one function built two different prompts. The second function forgot `return`, so it gave back `None`.

## Watch Out
- `print` only shows a value on screen. `return` hands the value back so other code can use it.
- Don't use a list as a default value (`history=[]`). All calls share that one list. Use `history=None` and create the list inside.

## Remember
- A function = **name + inputs + steps + result**.
- Use `return` to give a result back.
- Small functions are **easier to reuse and test**.

## Related Questions
- [What is object-oriented programming?](06-What-is-object-oriented-programming.md)
- [How do Python exceptions work?](08-How-do-Python-exceptions-work.md)
- [How do you handle API errors?](15-How-do-you-handle-API-errors.md)
