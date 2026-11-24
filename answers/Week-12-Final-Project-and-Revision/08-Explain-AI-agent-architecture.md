# Explain AI agent architecture.

> Week 12 – Final Project and Revision · Question 8

## In One Line
An AI agent is an LLM placed in a **loop**: it decides the next step, uses a tool, looks at the result, and repeats until the goal is done, with safety limits around it.

## Simple Explanation
A plain LLM call is one question → one answer. An **agent** can **take actions over several steps**. For example: "Refund order 123 if it's eligible" needs looking up the order, checking the policy, and then issuing the refund.

An agent has these parts:
- **The LLM (the brain):** decides what to do next.
- **Tools (the hands):** functions your code provides, like `get_order()`, `search_docs()`, `issue_refund()`.
- **The loop (the runtime):** your code that calls the LLM, runs the chosen tool, and feeds back the result.
- **Memory:** **short-term** (the messages and tool results in this task) and **long-term** (facts saved across sessions, e.g. in a vector database).
- **Guardrails:** step limits, allowed-tool lists, input checks, and **human approval** for risky actions.

```
Goal → LLM thinks → picks a tool → Safety check → Run tool → Result back to LLM → (repeat) → Final answer
```

**Step by step (refund example):**
1. The user gives the goal: "Refund order 123 if eligible."
2. The LLM decides it needs information and asks for `get_order(order_id="123")`.
3. Your code checks this tool is allowed, runs it, and gets: "bought 10 days ago, $40".
4. The result goes back to the LLM. It reasons: "the policy allows refunds within 30 days, so this is eligible."
5. The LLM asks for `issue_refund(order_id="123")`.
6. This is a **risky** tool, so your code **pauses and asks a human** to approve.
7. Once approved, your code runs it and sends back "refunded".
8. The LLM sees the goal is done and writes the final answer.
9. At every step, the runtime enforces **limits**: a maximum number of steps, a time limit, and a token budget, so the agent can't loop forever.

**Important:** the LLM only *asks* for tools. **Your code runs them.** That's where you enforce permissions.

**Common patterns:**
- **Single agent with tools** (often called ReAct: reason, then act): the simplest and most common.
- **Fixed workflow:** the steps are set in code, and the LLM fills in parts. This is more predictable; prefer it when you already know the steps.
- **Multi-agent:** several agents (e.g. planner, researcher, writer). This is powerful but costs more and has more ways to fail.

**Why care?** Agents can do real work, but mistakes add up over many steps, costs grow with each step, and actions have real effects.

## Real-Life Example
A travel agent (the LLM) with a phone (tools), a notepad for this trip (short-term memory), your customer file (long-term memory), and a manager who must sign off on any purchase over $500 (human approval).

## Watch Out
- Always set a **step limit and budget**. Without them, an agent can loop and burn money.
- Don't use an agent when a simple fixed workflow would do. More freedom means less predictability.

## Remember
- Agent = **LLM + tools + memory + loop + guardrails**.
- The LLM decides; **your code executes** and enforces permissions.
- Use **human approval** for risky actions and log every step.

## Related Questions
- [What is an AI agent?](../Week-10-AI-Agents-and-Tool-Calling/01-What-is-an-AI-agent.md)
- [What is an agent execution loop?](../Week-10-AI-Agents-and-Tool-Calling/11-What-is-an-agent-execution-loop.md)
- [What is agent memory?](../Week-10-AI-Agents-and-Tool-Calling/06-What-is-agent-memory.md)
- [What is a multi-agent system?](../Week-10-AI-Agents-and-Tool-Calling/10-What-is-a-multi-agent-system.md)
- [How do you implement human approval for sensitive operations?](../Week-11-Production-LLM-Applications/18-How-do-you-implement-human-approval-for-sensitive-operations.md)
- [How does function calling work?](09-How-does-function-calling-work.md)
