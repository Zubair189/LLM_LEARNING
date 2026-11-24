# What happens when a tool call fails?

> Week 10 – AI Agents and Tool Calling · Question 12

## In One Line
It depends on your code: a well-built agent **catches the error and tells the model about it**, so the model can try again, try something else, or explain the problem.

## Simple Explanation
Tools fail all the time: a website is down, a city name is misspelled, a file doesn't exist, or a request times out.

**In a badly built agent:**
- The error **crashes** the whole program, or
- The error is **silently ignored**, and the model may then **make up** a result (hallucinate).

**In a well-built agent:**
1. Your code wraps the tool in a `try/except` (Python's way of catching errors).
2. If it fails, you send the model a **tool result marked as an error**, with a clear message. With Anthropic this is `"is_error": True`, for example "City 'Pariss' not found. Did you mean 'Paris'?"
3. The model reads the error and decides what to do:
   - **retry** with fixed inputs,
   - **use a different tool**,
   - **ask the user** for help,
   - or **explain** honestly that it couldn't finish.

**Good habits:**
- Write **helpful error messages**: say what went wrong and how to fix it.
- **Retry automatically** for temporary problems like timeouts, but only a few times.
- **Limit total attempts**, so the agent doesn't keep retrying forever.
- **Log** every failure so you can fix the cause later.
- Be extra careful retrying actions like payments, so you don't charge someone twice.

## Real-Life Example
You ask an assistant to call a shop, but the number is wrong. A bad assistant says nothing, or invents "they said yes". A good assistant comes back and says "the number didn't work; should I look it up online?"

## Code Example
Catch a tool error and turn it into a message the model can read.

```python
WEATHER = {"Paris": "18°C", "Rome": "24°C"}

def get_weather(city):
    if city not in WEATHER:
        raise ValueError(f"City '{city}' not found. Known cities: Paris, Rome")
    return WEATHER[city]

def run_tool(tool_use_id, city):
    try:
        return {"type": "tool_result", "tool_use_id": tool_use_id,
                "content": get_weather(city)}                     # success
    except Exception as e:
        return {"type": "tool_result", "tool_use_id": tool_use_id,
                "content": str(e), "is_error": True}              # failure, told politely

print(run_tool("t1", "Paris"))
print(run_tool("t2", "Pariss"))

# {'type': 'tool_result', 'tool_use_id': 't1', 'content': '18°C'}
# {'type': 'tool_result', 'tool_use_id': 't2', 'content': "City 'Pariss' not found. Known cities: Paris, Rome", 'is_error': True}
```

**What happened:** the program didn't crash. The model would receive the second result, see the helpful message, and most likely retry with "Paris".

## Watch Out
- Never hide errors from the model. It may invent a result instead.
- Don't blindly retry actions that change things (like payments or emails).

## Remember
- Catch errors and send them back with `is_error: True`.
- Clear error messages help the model recover.
- Limit retries and log failures.

## Related Questions
- [What is an agent execution loop?](11-What-is-an-agent-execution-loop.md)
- [What is tool calling?](03-What-is-tool-calling.md)
- [What are human-in-the-loop systems?](13-What-are-human-in-the-loop-systems.md)
- [What are the limitations of autonomous agents?](15-What-are-the-limitations-of-autonomous-agents.md)
