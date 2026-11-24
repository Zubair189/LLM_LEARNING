# How does function calling work?

> Week 12 – Final Project and Revision · Question 9

## In One Line
Function calling lets the LLM **ask your code to run a function** (like "get the weather for Paris"); your code runs it and sends the result back so the model can finish its answer.

## Simple Explanation
An LLM can only write text. It can't check today's weather or look up an order by itself. **Function calling** (also called **tool use**) is the bridge.

You tell the model which tools exist. Each tool has:
- a **name**, e.g. `get_weather`,
- a **description** in plain words, e.g. "Get the current weather for a city",
- an **input schema**: a description, in **JSON Schema** format, of what inputs it needs, e.g. `city` (text).

The model reads these descriptions and, when a tool would help, replies with a structured **request** instead of normal text.

```
App sends question + tool list → LLM: "please call get_weather(city=Paris)" → App checks & runs it → App sends result back → LLM writes final answer
```

**Step by step:**
1. **Your app** sends the user's message ("What's the weather in Paris?") plus the list of tools.
2. **The model** decides a tool is needed. It stops with `stop_reason = "tool_use"` and returns a **tool_use block**: the tool name, the inputs (`{"city": "Paris"}`) and an **id**.
3. **Your app** checks the inputs: correct type? Allowed? Does this user have permission?
4. **Your app** runs the real function and gets a result, e.g. `{"temp_c": 18, "sky": "sunny"}`.
5. **Your app** sends the conversation back, adding the model's tool request and a **tool_result** that carries the same **id**, so the model knows which request it answers.
6. **The model** reads the result and writes the final answer: "It's 18°C and sunny in Paris." This time `stop_reason` is `"end_turn"`.

If the model asks for another tool, you repeat steps 3–5. An **agent** is just this loop repeated many times (see Q8).

**How does the model choose?** It was trained to produce tool requests, and it picks based on your **descriptions**. Clear names and descriptions are basically prompt engineering for tools.

**A handy side use:** if you *force* the model to use one tool, you get its answer as clean structured JSON, which is great for pulling fields out of an invoice, for example.

**Safety rules:**
- Treat the model's inputs as **untrusted**. Always validate them.
- Keep tools **narrow**: `get_order(id)` is safer than `run_any_sql(query)`.
- Ask a **human** before risky tools run, and log every call.
- If a tool fails, send back an error result so the model can explain or try something else.

## Real-Life Example
A manager (the LLM) fills in a request form: "Please fetch the Paris weather report." They hand it to an assistant (your code). The assistant checks the form, does the task, and brings back the report. The manager then writes the reply. The manager never touches the filing cabinet directly.

## Watch Out
- **The LLM never runs the function itself.** It only asks; your code decides and runs it.
- If you forget to send the `tool_result` with the matching id, the conversation breaks.

## Remember
- You describe tools; the model **requests** a call; **your code runs it** and returns the result.
- `stop_reason == "tool_use"` means "run a tool"; `"end_turn"` means "done".
- Validate inputs and keep tools small and safe.

## Related Questions
- [What is function calling?](../Week-10-AI-Agents-and-Tool-Calling/04-What-is-function-calling.md)
- [What is tool calling?](../Week-10-AI-Agents-and-Tool-Calling/03-What-is-tool-calling.md)
- [How does an LLM decide which tool to use?](../Week-10-AI-Agents-and-Tool-Calling/05-How-does-an-LLM-decide-which-tool-to-use.md)
- [What happens when a tool call fails?](../Week-10-AI-Agents-and-Tool-Calling/12-What-happens-when-a-tool-call-fails.md)
- [What is structured output?](../Week-06-Working-with-LLMs-and-Model-APIs/15-What-is-structured-output.md)
- [Explain AI agent architecture.](08-Explain-AI-agent-architecture.md)
