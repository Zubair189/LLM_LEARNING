# What is an AI agent?

> Week 10 – AI Agents and Tool Calling · Question 1

## In One Line
An AI agent is an LLM that works **in a loop toward a goal**: it decides what to do, uses tools, looks at the results, and keeps going until the job is done.

## Simple Explanation
A normal chatbot answers one message and stops. An agent can take **several steps** and **act** in the world.

An agent is built from a few parts:
- **The LLM (the brain):** decides the next step.
- **Tools (the hands):** things it can use, like web search, a calculator, a database, sending an email, or running code.
- **The loop:** your program keeps asking the LLM "what next?" and runs the tools it asks for.
- **Memory:** what it has seen so far, plus saved notes.
- **Guardrails (safety rules):** limits on what it may do and how many steps it may take.

The loop goes like this:

1. **Think:** the LLM looks at the goal and decides on an action.
2. **Act:** your code runs a tool, for example "search flights to Rome".
3. **Observe:** the tool's result goes back to the LLM.
4. **Repeat** until the LLM says "done", or a step limit is reached.

**Why it matters:** agents can finish real tasks, such as booking, researching or fixing code, instead of only giving advice.

## Real-Life Example
Asking a friend "what's the weather in Paris?" gets you an answer. Hiring a travel assistant to "plan my trip to Paris" is different: they check the weather, look up hotels, compare prices, and come back with a plan. The travel assistant is like an agent.

## Code Example
A toy agent with a fake "brain" and fake tools, so it runs without any AI.

```python
tools = {
    "get_weather": lambda city: f"{city}: 18°C, rain",   # fake weather tool
    "pack":        lambda item: f"packed {item}",         # fake packing tool
}

def fake_llm(history):            # stands in for the real LLM's decision
    if len(history) == 0: return ("get_weather", "Paris")
    if len(history) == 1: return ("pack", "umbrella")
    return ("done", "Ready: it's rainy, umbrella packed.")

history = []
for step in range(5):             # loop, with a step limit for safety
    action, arg = fake_llm(history)            # 1. think
    if action == "done":
        print("Final:", arg); break
    result = tools[action](arg)                # 2. act
    history.append(result)                     # 3. observe
    print(f"Step {step + 1}: {action}({arg}) -> {result}")

# Step 1: get_weather(Paris) -> Paris: 18°C, rain
# Step 2: pack(umbrella) -> packed umbrella
# Final: Ready: it's rainy, umbrella packed.
```

**What happened:** the "brain" chose an action, the program ran the tool, and the result fed the next decision. A real agent swaps `fake_llm` for an LLM API call.

## Watch Out
- The LLM doesn't run tools itself. **Your code** runs them and sends results back.
- More freedom also means more risk. Real agents need step limits and safety checks.

## Remember
- Agent = LLM + tools + loop (+ memory and guardrails).
- Think → act → observe → repeat.
- It can take many steps toward a goal.

## Related Questions
- [What is the difference between an LLM and an AI agent?](02-What-is-the-difference-between-an-LLM-and-an-AI-agent.md)
- [What is an agent execution loop?](11-What-is-an-agent-execution-loop.md)
- [What is tool calling?](03-What-is-tool-calling.md)
- [What is the Model Context Protocol (MCP)?](16-BONUS-What-is-the-Model-Context-Protocol-MCP.md)
