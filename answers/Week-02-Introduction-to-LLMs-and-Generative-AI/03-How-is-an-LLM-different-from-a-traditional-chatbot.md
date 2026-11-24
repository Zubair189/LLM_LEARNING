# How is an LLM different from a traditional chatbot?

> Week 02 – Introduction to LLMs and Generative AI · Question 3

## In One Line
A traditional chatbot follows a **fixed script**, while an LLM **writes each reply fresh**, so it can handle questions nobody planned for.

## Simple Explanation
**Traditional chatbot:**
1. It looks for keywords or known topics in your message (for example "refund").
2. It picks a ready-made reply written by a person.
3. If your message doesn't match, it says "Sorry, I didn't understand."

It is **predictable and cheap**, but it breaks easily. Say "money back" instead of "refund" and it may fail.

**LLM-based assistant:**
1. It reads your whole message plus the conversation so far.
2. It **generates** a new reply, word by word, from patterns learned from huge amounts of text.

It understands different wordings and follow-up questions, and it can do many tasks. But it is **less predictable**, can **hallucinate** (say false things confidently), and costs more per message.

That is why real products often **mix both**:
- The LLM **understands** what the user wants.
- Normal code **does** the important actions, like issuing a refund, safely and exactly.

Also note: "chatbot" is the **interface**, the chat window. "LLM" is the **engine**. An LLM can power many things besides chat.

## Real-Life Example
A traditional chatbot is like a phone menu: "Press 1 for billing, press 2 for support." An LLM is like a helpful human operator who understands "I was charged twice last Tuesday and I'm also moving house. What should I do?"

## Code Example
A rule-based bot, and where it breaks.

```python
RULES = {                                       # keyword -> ready-made reply
    "hours": "We are open 9am-5pm, Monday to Friday.",
    "refund": "To request a refund, visit /refunds.",
}

def rule_bot(message):
    for keyword, reply in RULES.items():        # check each keyword
        if keyword in message.lower():          # found it?
            return reply                        # give the canned answer
    return "Sorry, I didn't understand that."   # no match: give up

print(rule_bot("What are your hours?"))
print(rule_bot("Can I get my money back?"))
# Output:
# We are open 9am-5pm, Monday to Friday.
# Sorry, I didn't understand that.
```

**What happened:** "money back" means "refund", but the bot only knows exact keywords, so it failed. An LLM, given the same refund info in its instructions, would understand and point the user to /refunds.

## Watch Out
- **LLMs don't replace rules everywhere.** Payments and permissions should still go through exact, tested code.
- **An LLM doesn't know your company's data** unless you give it to the model, in the prompt or through tools.

## Remember
- Traditional bot = fixed script, predictable, brittle.
- LLM = generates replies, flexible, can be wrong.
- Best systems combine both.

## Related Questions
- [What is a Large Language Model (LLM)?](./02-What-is-a-Large-Language-Model-LLM.md)
- [Why do LLMs sometimes hallucinate?](./12-Why-do-LLMs-sometimes-hallucinate.md)
- [What is the difference between an LLM and an AI agent?](../Week-10-AI-Agents-and-Tool-Calling/02-What-is-the-difference-between-an-LLM-and-an-AI-agent.md)
- [How do you maintain conversation history?](../Week-11-Production-LLM-Applications/04-How-do-you-maintain-conversation-history.md)
