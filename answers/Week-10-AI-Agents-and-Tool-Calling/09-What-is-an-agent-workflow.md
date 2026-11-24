# What is an agent workflow?

> Week 10 – AI Agents and Tool Calling · Question 9

## In One Line
An agent workflow is a **fixed series of steps** (LLM calls, tool calls and normal code) that **you**, the developer, design in advance to get a job done.

## Simple Explanation
There are two ways to build an AI system that does multi-step work:

- **Workflow:** *you* decide the steps and their order. The LLM does the thinking inside each step, but it doesn't choose what comes next.
- **Autonomous agent:** the *LLM* decides the next step each time, in a loop.

Common workflow patterns:
1. **Chain:** step 1 → step 2 → step 3. For example: write an outline → write the draft → check the grammar.
2. **Router:** first sort the request, then send it down the right path, such as "billing question" → billing steps.
3. **Parallel:** run several LLM calls at the same time and combine them, such as checking a text for tone, facts and length all at once.
4. **Checker loop:** one LLM writes, another reviews; repeat until it passes (with a maximum number of tries).

**Why most real systems use workflows:**
- **Predictable:** the same input follows the same path.
- **Easy to test and debug:** you know exactly which step failed.
- **Cheaper and faster:** no extra "what should I do next?" calls.

Use a full agent only when the steps really can't be known in advance, for example open-ended research or fixing an unknown bug. Many products are workflows with a small agent inside one step.

## Real-Life Example
A workflow is like a recipe: chop, then fry, then season, always in that order. An autonomous agent is like a chef told "make something good with what's in the fridge". They decide the steps themselves. Recipes are more reliable; the free chef is more flexible.

## Code Example
A simple chain workflow with a router step. The "LLM" functions are fake so it runs anywhere.

```python
def classify(text):                    # step 1: an LLM would sort the request
    return "billing" if "refund" in text else "general"

def billing_reply(text):               # step 2a: billing path
    return "Your refund request has been logged."

def general_reply(text):               # step 2b: general path
    return "Thanks! Here's some general help."

def workflow(text):                    # the developer fixed this order
    category = classify(text)
    reply = billing_reply(text) if category == "billing" else general_reply(text)
    return f"[{category}] {reply}"

print(workflow("I want a refund"))     # [billing] Your refund request has been logged.
print(workflow("How do I log in?"))    # [general] Thanks! Here's some general help.
```

**What happened:** the path was chosen by our code (the router), not by an LLM deciding freely. In a real app, `classify` and the reply steps would each be an LLM call.

## Watch Out
- "Agent" is often used loosely. Ask: who decides the next step, the code or the model?
- Don't build a full agent when a simple chain of steps would work.

## Remember
- Workflow = developer decides the steps; agent = model decides.
- Patterns: chain, router, parallel, checker loop.
- Workflows are more predictable, cheaper and easier to test.

## Related Questions
- [What is an agent execution loop?](11-What-is-an-agent-execution-loop.md)
- [What is a multi-agent system?](10-What-is-a-multi-agent-system.md)
- [What are human-in-the-loop systems?](13-What-are-human-in-the-loop-systems.md)
- [What is an AI agent?](01-What-is-an-AI-agent.md)
