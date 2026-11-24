# What are the limitations of autonomous agents?

> Week 10 – AI Agents and Tool Calling · Question 15

## In One Line
Autonomous agents are still **unreliable on long tasks**: small mistakes add up, they can get stuck or tricked, and they are slow, costly and hard to test.

## Simple Explanation
"Autonomous" means the agent decides its own steps with little human help. That's powerful, but it has real limits today:

1. **Errors add up.** If each step is 95% reliable, a 20-step task succeeds only about 36% of the time. One wrong step early can ruin everything after it.
2. **Making things up.** The model can **hallucinate**, meaning it invents facts, or even claims a tool succeeded when it didn't.
3. **Getting stuck or wandering.** It may repeat the same action in a loop, or drift away from the real goal.
4. **Prompt injection.** Text the agent reads (web pages, emails) can contain hidden instructions that trick it.
5. **Cost and speed.** Many LLM calls per task means more money and waiting.
6. **Hard to test.** The same request can take a different path each time, so bugs are hard to reproduce.
7. **Limited memory.** Long tasks can overflow the context window, so it forgets earlier details.

**How teams deal with this today:** they keep agents **bounded**:
- only the tools they need,
- step and budget limits,
- sandboxes for running code,
- human approval for important actions,
- and good logging and testing.

Often a simple **workflow** with a small agent inside is the better choice.

## Real-Life Example
Imagine sending a very clever but brand-new intern alone on a 20-errand trip across the city, with no phone check-ins. They'll probably do most errands well, but one wrong turn early can throw off the whole day. So you give them a list, a budget, and ask them to call before any big purchase.

## Code Example
See how small per-step errors add up over many steps.

```python
step_success = 0.95                       # each step works 95% of the time

for steps in [1, 5, 10, 20, 50]:
    whole_task = step_success ** steps    # every step must succeed
    print(f"{steps:>2} steps -> {whole_task:.0%} chance of success")

#  1 steps -> 95% chance of success
#  5 steps -> 77% chance of success
# 10 steps -> 60% chance of success
# 20 steps -> 36% chance of success
# 50 steps -> 8% chance of success
```

**What happened:** even a "95% good" agent fails most long tasks if every step must be right. That's why checks, retries and human review matter. (Real agents can sometimes notice and fix their own mistakes, which helps a bit.)

## Watch Out
- A great demo doesn't mean reliable daily use. Test on many real tasks.
- More autonomy means more risk. Give agents only as much freedom as the task needs.

## Remember
- Small errors compound over many steps.
- Agents can hallucinate, loop, or be tricked by prompt injection.
- Keep agents bounded: limits, sandboxes, human approval.

## Related Questions
- [How do you prevent an agent from performing unauthorized operations?](14-How-do-you-prevent-an-agent-from-performing-unauthorized-operations.md)
- [What are human-in-the-loop systems?](13-What-are-human-in-the-loop-systems.md)
- [What is a multi-agent system?](10-What-is-a-multi-agent-system.md)
- [How do you evaluate a fine-tuned model?](../Week-09-Fine-tuning-and-Model-Optimization/14-How-do-you-evaluate-a-fine-tuned-model.md)
