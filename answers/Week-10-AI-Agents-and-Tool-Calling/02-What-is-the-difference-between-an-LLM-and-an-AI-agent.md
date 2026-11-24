# What is the difference between an LLM and an AI agent?

> Week 10 – AI Agents and Tool Calling · Question 2

## In One Line
An LLM is the **engine** that turns text into text; an AI agent is the **whole car** built around it, with tools, a loop, memory and safety rules so it can actually do things.

## Simple Explanation
**An LLM on its own:**
- Takes text in, gives text out, in **one call**.
- Remembers nothing between calls (it is **stateless**). Any "memory" must be sent back in with the next message.
- Cannot do anything in the world. It can only write.

**An AI agent** is a program built around an LLM. It adds:
1. **Tools**, so it can search, calculate, read files or send emails.
2. **A loop**, so it can take many steps: ask the LLM, run a tool, give back the result, ask again.
3. **Memory**, so it keeps track of what has happened.
4. **Guardrails**, so it has limits on actions, steps and cost.

**An important detail:** modern LLMs can *write* a request like "please call `get_weather` for Paris". That alone does not make an agent. It only becomes an agent when **your code** actually runs that tool, sends the result back, and repeats.

**Which should you use?** For one-shot jobs like summarising or translating, a single LLM call is simpler, cheaper and more predictable. Use an agent when the task needs several steps or live information.

## Real-Life Example
An LLM is like a very knowledgeable expert you can phone. They give great answers, but they can't go anywhere or press any buttons. An agent is like that expert sitting at a desk with a computer, a phone and a to-do list. They can look things up, do the steps, and report back when finished.

## Code Example
One LLM call versus an agent loop (both use fake functions so they run anywhere).

```python
def llm(prompt):                      # fake LLM: text in, text out, nothing else
    return "I can't check live prices, but laptops usually cost $500-$1500."

print("LLM:", llm("Price of laptop X?"))   # one answer, no actions

def price_tool(item):                 # a real-world action the agent can take
    return 899

def agent(goal):
    price = price_tool("laptop X")    # step 1: use a tool
    if price < 1000:                  # step 2: decide based on the result
        return f"Laptop X costs ${price}, which is under budget."
    return "Too expensive."

print("Agent:", agent("Is laptop X under $1000?"))

# LLM: I can't check live prices, but laptops usually cost $500-$1500.
# Agent: Laptop X costs $899, which is under budget.
```

**What happened:** the plain LLM could only guess. The agent used a tool to get a real fact and acted on it. In a real agent, the LLM (not fixed `if` rules) decides which step comes next.

## Watch Out
- Calling an LLM "an agent" just because it chats is a common mix-up.
- Agents are slower, costlier and harder to test than single calls. Don't use one when one call will do.

## Remember
- LLM = text in, text out, one call.
- Agent = LLM + tools + loop + memory + guardrails.
- Your code runs the tools, not the LLM.

## Related Questions
- [What is an AI agent?](01-What-is-an-AI-agent.md)
- [What is an agent execution loop?](11-What-is-an-agent-execution-loop.md)
- [What is agent memory?](06-What-is-agent-memory.md)
- [What are the limitations of autonomous agents?](15-What-are-the-limitations-of-autonomous-agents.md)
