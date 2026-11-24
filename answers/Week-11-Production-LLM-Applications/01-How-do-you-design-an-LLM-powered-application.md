# How do you design an LLM-powered application?

> Week 11 – Production LLM Applications · Question 1

## In One Line
You design an LLM app by treating the model as **one part** of a bigger system, and building safe, cheap and measurable steps around it.

## Simple Explanation
An **LLM** (large language model) is the "brain" that writes answers. But a real app needs much more than a brain. The model can make mistakes, and every call costs money, so the code around it keeps things safe and under control.

A simple way to design one:

1. **Decide what "good" looks like.** Write down 20–50 example questions with the answers you expect. This becomes your **test set** (a fixed list you check the app against).
2. **Pick the simplest pattern that works.**
   - Just a prompt: for summarising, rewriting or sorting text.
   - **RAG** (the app looks up your own documents first): when answers must come from your data.
   - **Tools**: when the model must fetch live data or do actions.
   - **Agent** (a loop of many steps): only when the steps are not known in advance.
3. **Put a small service in front of the model.** It checks the input, builds the prompt, calls the model and checks the output.
4. **Add the "production" parts:** login, **rate limits** (a cap on how often people can call it), caching, logging and guardrails.
5. **Test again after every change** using your test set.

The flow looks like this:

```
User → API (checks) → Cache? → (Documents / Tools) → LLM → Output checks → User
                                                       ↘ Logs & cost tracking
```

## Real-Life Example
Think of a restaurant. The chef (the LLM) is important, but you don't let the chef stand at the door. A host checks bookings, a waiter takes orders, and someone checks each plate before it goes out. The whole restaurant works together, not just the chef.

## Code Example
A tiny "app" with input checks, a cache, a model call and a log line.

```python
def fake_llm(prompt):                       # stand-in for the real model call
    return "RAG means the model looks up documents before answering."

cache = {}                                  # remembers past answers

def handle(question):
    if not question or len(question) > 2000:   # 1. check the input
        return "Error: question is empty or too long"
    if question in cache:                      # 2. already answered? reuse it
        return cache[question] + " (from cache)"
    prompt = "You are a helpful tutor.\nQuestion: " + question   # 3. build prompt
    answer = fake_llm(prompt)                  # 4. ask the model
    cache[question] = answer                   # 5. save for next time
    print("log: answered", len(question), "chars")   # 6. keep a record
    return answer

print(handle("What is RAG?"))
print(handle("What is RAG?"))
print(handle(""))
# Output:
# log: answered 12 chars
# RAG means the model looks up documents before answering.
# RAG means the model looks up documents before answering. (from cache)
# Error: question is empty or too long
```

**What happened:** the model was called only once. The second question came from the cache, and the empty question was stopped before it could cost anything.

## Watch Out
- Don't start with a complex agent when one simple prompt would do.
- Never trust the model's output blindly, for example by running code or SQL it wrote without checking it.

## Remember
- The LLM is **one part**; the system around it makes it safe.
- Start with the **simplest pattern** that solves the problem.
- Build in checks, caching, logs and a test set **from day one**.

## Related Questions
- [What is the role of an API gateway?](02-What-is-the-role-of-an-API-gateway.md)
- [How does FastAPI help in AI applications?](03-How-does-FastAPI-help-in-AI-applications.md)
- [What is a RAG pipeline?](../Week-08-Retrieval-Augmented-Generation-RAG/11-What-is-a-RAG-pipeline.md)
- [How do you deploy an LLM application in production?](../Week-12-Final-Project-and-Revision/12-How-do-you-deploy-an-LLM-application-in-production.md)
