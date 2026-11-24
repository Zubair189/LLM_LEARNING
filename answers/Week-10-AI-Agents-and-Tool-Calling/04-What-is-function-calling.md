# What is function calling?

> Week 10 – AI Agents and Tool Calling · Question 4

## In One Line
Function calling is when an LLM, instead of plain text, outputs a **neat, structured request** to run one of your functions, such as `get_weather(city="Paris")`.

## Simple Explanation
**Tool calling vs. function calling:** they mean basically the same thing; "function calling" is the older name (from OpenAI in 2023) and "tool calling" is the broader, modern one.

Here's how it works:

1. You write a normal function in your code, like `get_weather(city)`.
2. You describe it to the model: its name, what it does, and its inputs.
3. When useful, the model outputs a structured request, usually **JSON** (a simple text format for data), like `{"name": "get_weather", "arguments": {"city": "Paris"}}`.
4. Your code reads that JSON, finds the matching function, and runs it.

Why is a structured request useful? Because your program can read it reliably. Free text like "maybe check Paris weather?" is hard for code to understand. Clean JSON is easy.

**A handy extra use: data extraction.** You can define a "function" just to get tidy data out of messy text. For example, a `save_contact(name, email)` function makes the model pull the name and email out of an email signature in a fixed format, even if you never really "call" anything.

## Real-Life Example
Ordering at a restaurant with a printed order form instead of shouting your order. The form has fixed boxes: dish, size, extras. The kitchen can read a filled-in form instantly, with no guessing.

## Code Example
Turn a model's structured output into a real function call. The model's reply is faked here so it runs anywhere.

```python
import json

def get_weather(city):                       # your real function
    return f"{city}: 18°C, light rain"

functions = {"get_weather": get_weather}     # the functions the model may call

# Pretend this JSON came back from the model
model_output = '{"name": "get_weather", "arguments": {"city": "Paris"}}'

request = json.loads(model_output)           # turn JSON text into a Python dict
func = functions[request["name"]]            # look up the function by name
result = func(**request["arguments"])        # run it with the given inputs

print(result)   # Paris: 18°C, light rain
```

**What happened:** the model only produced a structured request. Our code read it, picked the right function, and ran it. In a real app, you then send the result back to the model.

## Watch Out
- The model doesn't run the function; your code does. Always check the function name and inputs first.
- Different providers use different names and formats, but the idea is the same.

## Remember
- Function calling ≈ tool calling. Use whichever term your provider uses.
- The model outputs a structured request (JSON); your code runs it.
- It's also great for pulling tidy data out of messy text.

## Related Questions
- [What is tool calling?](03-What-is-tool-calling.md)
- [How does an LLM decide which tool to use?](05-How-does-an-LLM-decide-which-tool-to-use.md)
- [What is the Model Context Protocol (MCP)?](16-BONUS-What-is-the-Model-Context-Protocol-MCP.md)
