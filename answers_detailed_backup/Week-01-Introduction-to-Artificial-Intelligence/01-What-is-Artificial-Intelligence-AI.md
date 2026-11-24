# What is Artificial Intelligence (AI)?

> Week 01 – Introduction to Artificial Intelligence · Question 1

## Short Answer
Artificial Intelligence is the field of computer science that builds systems able to do tasks we normally associate with human intelligence, such as understanding language, recognizing images, making decisions, and planning. AI is an umbrella term. It covers hand-written rule systems as well as systems that learn from data (machine learning). Today's most visible AI, such as large language models like Claude or GPT, is built with deep learning.

## Detailed Explanation
"Intelligence" here means behaviour, not consciousness. We call a system AI if it can **perceive**, **reason**, **learn**, or **act** in ways that would need intelligence if a person did them.

**Two broad families of approaches:**

| Approach | How it works | Example |
|---|---|---|
| Symbolic / rule-based AI ("GOFAI") | Humans write explicit rules and logic | Chess engines of the 1990s, expert systems, `if fever and cough: flu` |
| Machine learning (data-driven) | The system learns patterns from examples | Spam filters, image recognition, LLMs |

**The nesting diagram you should memorize:**

```
┌──────────────────────────── Artificial Intelligence ──────────────┐
│  rule-based systems, search, planning, ...                        │
│  ┌──────────────────────── Machine Learning ──────────────────┐   │
│  │  decision trees, linear models, clustering, ...            │   │
│  │  ┌───────────────────── Deep Learning ─────────────────┐   │   │
│  │  │  neural networks with many layers                   │   │   │
│  │  │  ┌──────── Generative AI / LLMs ───────┐            │   │   │
│  │  │  │  Claude, GPT, Gemini, Llama, ...    │            │   │   │
│  │  │  └─────────────────────────────────────┘            │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  └────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
```

**Narrow vs. general AI:**
- **Narrow AI**: very good at one task, such as translation or face detection. Every deployed system in 2026 is narrow in the strict sense, though LLMs are much more general-purpose than earlier systems.
- **AGI (Artificial General Intelligence)**: a hypothetical system that matches humans across essentially all cognitive tasks. People still disagree about how to define it and when it might arrive.

**A short history:**
- 1950s–80s: symbolic AI, logic, expert systems.
- 1990s–2000s: statistical machine learning.
- 2012 onward: deep learning (ImageNet).
- 2017: the Transformer architecture.
- 2020s: large language models and generative AI.

## Example
**Analogy:** You can teach someone to make tea in two ways. You can write a precise recipe (rule-based AI), or you can let them watch you make tea 1,000 times until they work it out (machine learning). Both count as "AI" if the result is a competent tea-maker.

**Python: a tiny rule-based "AI" agent (no learning):**
```python
def thermostat_agent(temp_c: float) -> str:
    """Perceive temperature -> decide -> act. Intelligent behaviour, zero learning."""
    if temp_c < 18:
        return "heat on"
    elif temp_c > 25:
        return "cooling on"
    return "idle"

for t in [15, 21, 30]:
    print(t, "->", thermostat_agent(t))
# 15 -> heat on
# 21 -> idle
# 30 -> cooling on
```
This is AI in the classic "agent" sense: it perceives, decides, and acts. It is **not** machine learning, because a human wrote every rule.

## Common Mistakes / Misconceptions
- **"AI = ChatGPT."** LLMs are one small, recent part of AI. Route planners, recommender systems and fraud detectors are AI too.
- **"AI understands like humans."** Current systems are statistical pattern learners. They can behave intelligently without human-like understanding or awareness.
- **"AI always learns."** Rule-based systems are AI but learn nothing.
- **"AI is objective."** Systems inherit biases from their data and from their designers' choices.

## Key Takeaways
- AI is the umbrella field for machines that perform tasks needing intelligence.
- Its two broad families are rule-based (hand-coded) and learning-based (machine learning).
- The nesting is AI ⊃ ML ⊃ Deep Learning ⊃ LLMs / Generative AI.
- All practical AI today is task-oriented ("narrow"); AGI is still a debated goal.

## Related Questions
- [What is the difference between AI and Machine Learning?](./02-What-is-the-difference-between-AI-and-Machine-Learning.md)
- [What is Deep Learning?](./07-What-is-Deep-Learning.md)
- [What is Generative AI?](../Week-02-Introduction-to-LLMs-and-Generative-AI/01-What-is-Generative-AI.md)
- [What is a Large Language Model (LLM)?](../Week-02-Introduction-to-LLMs-and-Generative-AI/02-What-is-a-Large-Language-Model-LLM.md)
