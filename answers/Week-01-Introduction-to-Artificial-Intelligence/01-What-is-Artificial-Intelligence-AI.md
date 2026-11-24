# What is Artificial Intelligence (AI)?

> Week 01 – Introduction to Artificial Intelligence · Question 1

## In One Line
Artificial Intelligence (AI) means building computer systems that can do tasks which normally need human thinking, like understanding language, recognising pictures or making decisions.

## Simple Explanation
"Intelligence" here is about **behaviour**, not feelings or awareness. If a machine can notice what is going on, make a sensible decision and act on it, we call it AI.

There are two main ways to build AI:

1. **Rule-based AI:** people write the rules by hand.
   *Example:* "if the temperature is below 18 °C, turn the heater on".
2. **Machine Learning (ML):** the computer learns the rules by itself from many examples.
   *Example:* a spam filter that learns from thousands of emails.

AI is an **umbrella word**. Inside it sit smaller ideas, like boxes inside boxes:

**AI ⊃ Machine Learning ⊃ Deep Learning ⊃ LLMs**

(⊃ means "contains". Every LLM is deep learning, every deep learning system is ML, and every ML system is AI.)

All AI we use today is **narrow AI**: it is built for certain kinds of tasks. **AGI** (Artificial General Intelligence) would be a system as capable as a human at almost everything. It does not exist yet, and people still argue about how to define it.

**Link to LLMs:** chat assistants like Claude are one recent, very visible kind of AI. They are built with deep learning.

## Real-Life Example
You can teach someone to make tea in two ways. You can hand them an exact recipe (rule-based AI). Or you can let them watch you make tea a thousand times until they work it out (machine learning). Either way, you end up with a good tea-maker.

## Code Example
A tiny rule-based "AI" thermostat. It notices, decides and acts, but it does not learn.

```python
def thermostat(temp_c):              # input: the room temperature
    if temp_c < 18:                  # too cold?
        return "heat on"
    elif temp_c > 25:                # too hot?
        return "cooling on"
    return "idle"                    # just right: do nothing

for t in [15, 21, 30]:               # try three temperatures
    print(t, "->", thermostat(t))

# Output:
# 15 -> heat on
# 21 -> idle
# 30 -> cooling on
```

**What happened:** the program made a sensible decision for each temperature. It counts as simple AI, but it is **not** machine learning, because a human wrote every rule.

## Watch Out
- **"AI = ChatGPT."** Chatbots are only one small part of AI. Map route planners, film recommendations and fraud alerts are AI too.
- **"AI thinks like a human."** Today's AI finds patterns in data. It can act cleverly without truly understanding like a person.

## Remember
- AI = machines doing tasks that need "thinking".
- Two families: **hand-written rules** and **learning from data** (ML).
- AI ⊃ ML ⊃ Deep Learning ⊃ LLMs.

## Related Questions
- [What is the difference between AI and Machine Learning?](./02-What-is-the-difference-between-AI-and-Machine-Learning.md)
- [What is Deep Learning?](./07-What-is-Deep-Learning.md)
- [What is Generative AI?](../Week-02-Introduction-to-LLMs-and-Generative-AI/01-What-is-Generative-AI.md)
- [What is a Large Language Model (LLM)?](../Week-02-Introduction-to-LLMs-and-Generative-AI/02-What-is-a-Large-Language-Model-LLM.md)
