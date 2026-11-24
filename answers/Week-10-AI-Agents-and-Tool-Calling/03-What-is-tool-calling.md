# What is tool calling?

> Week 10 – AI Agents and Tool Calling · Question 3

## In One Line
Tool calling is how an LLM **asks your program to run something for it**, like checking the weather, and then uses the result in its answer.

## Simple Explanation
An LLM can only produce text. It can't look up today's weather or send an email by itself. Tool calling fixes that:

1. **You describe your tools** to the model. Each has a **name**, a **description** (what it does and when to use it), and an **input schema**, a small form that says which inputs it needs (for example, "city: text").
2. **The user asks something**, like "What's the weather in Paris?"
3. **The model replies with a tool request** instead of a normal answer: "call `get_weather` with city = Paris".
4. **Your code runs the tool** and gets the result, for example "18°C, light rain".
5. **You send the result back** to the model.
6. **The model writes the final answer**: "It's 18°C and rainy in Paris, so bring an umbrella."

The model **never runs the tool itself**. It only asks. Your program decides whether to run it, and does the actual work.

**Tool calling vs. function calling:** they mean basically the same thing; "function calling" is the older name and "tool calling" is the broader, modern one.

## Real-Life Example
A manager (the LLM) writes a note: "Please get me the sales numbers for March." An assistant (your code) goes to the filing cabinet, gets the numbers, and hands them back. The manager then writes the report. The manager never touches the cabinet.

## Code Example
Tool calling with the Anthropic Python SDK (needs an API key, so it is not run here).

```python
import anthropic
client = anthropic.Anthropic()

tools = [{
    "name": "get_weather",                                   # tool name
    "description": "Get the current weather for a city.",   # when to use it
    "input_schema": {"type": "object",                       # the inputs it needs
                     "properties": {"city": {"type": "string"}},
                     "required": ["city"]},
}]
messages = [{"role": "user", "content": "Weather in Paris?"}]

resp = client.messages.create(model="claude-sonnet-5-5", max_tokens=1024,
                              tools=tools, messages=messages)
call = next(b for b in resp.content if b.type == "tool_use")  # the model's request
print(call.name, call.input)          # e.g. get_weather {'city': 'Paris'}

messages += [{"role": "assistant", "content": resp.content},  # keep the request
             {"role": "user", "content": [{"type": "tool_result",
                 "tool_use_id": call.id, "content": "18°C, light rain"}]}]  # our result
final = client.messages.create(model="claude-sonnet-5-5", max_tokens=1024,
                               tools=tools, messages=messages)
print(final.content[0].text)          # e.g. "It's 18°C with light rain in Paris."
```

**What happened:** the model asked for `get_weather`, we "ran" it and sent the result back with the matching `tool_use_id`, and the model used it to answer.

## Watch Out
- A vague tool description leads to wrong or missing tool calls.
- Treat the model's tool inputs like user input: check them before running anything.

## Remember
- The model **asks**; your code **runs**.
- Each tool has a name, a description and an input schema.
- Results go back to the model so it can finish the answer.

## Related Questions
- [What is function calling?](04-What-is-function-calling.md)
- [How does an LLM decide which tool to use?](05-How-does-an-LLM-decide-which-tool-to-use.md)
- [What happens when a tool call fails?](12-What-happens-when-a-tool-call-fails.md)
- [What is the Model Context Protocol (MCP)?](16-BONUS-What-is-the-Model-Context-Protocol-MCP.md)
