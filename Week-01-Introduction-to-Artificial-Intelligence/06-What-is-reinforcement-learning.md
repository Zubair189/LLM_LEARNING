# What is reinforcement learning?

> Week 01 – Introduction to Artificial Intelligence · Question 6

## Short Answer
Reinforcement learning (RL) is a type of machine learning in which an **agent** learns by **trial and error**. It takes **actions** in an **environment** and receives **rewards** or penalties. Nobody shows it the correct action. Its goal is to learn a **policy**, a strategy for choosing actions, that maximizes total reward over time. RL powers game-playing systems like AlphaGo and robotics. It is also key to modern LLMs, through RLHF and reinforcement learning on verifiable tasks such as math and code.

## Detailed Explanation
**The RL loop:**
```
        ┌──────────── action a_t ─────────────┐
        │                                      ▼
   ┌─────────┐                          ┌─────────────┐
   │  Agent  │                          │ Environment │
   │ (policy)│◄── state s_t+1, reward r ─│             │
   └─────────┘                          └─────────────┘
```

**Core vocabulary:**
| Term | Meaning | Chess example |
|---|---|---|
| Agent | The learner / decision-maker | The chess program |
| Environment | The world it acts in | The board and the opponent |
| State | The current situation | Board position |
| Action | A choice the agent can make | A legal move |
| Reward | Feedback signal | +1 win, −1 loss, 0 otherwise |
| Policy | Mapping from state to action | The program's strategy |

**What makes RL different:**
- **Delayed reward:** a move made now may only pay off 40 moves later. This is the *credit assignment* problem.
- **Exploration vs. exploitation:** should the agent try something new, or repeat what has worked so far?
- **The data depends on the agent:** what it experiences depends on what it does, unlike a fixed labelled dataset.

**RL in LLMs (preview of the Week 2 bonus question):**
- **RLHF**: humans rank model answers, a *reward model* learns those preferences, and the LLM is optimized to produce answers the reward model scores highly.
- **RL with verifiable rewards**: for math or code, the reward is "did the answer or the tests pass?". This is how "reasoning" models are trained to think step by step.

## Example
**Analogy:** Training a dog. You don't explain the rules. You give a treat when it sits on command. Over many tries the dog learns which actions earn treats.

**Python: a multi-armed bandit with epsilon-greedy exploration (simplest RL):**
```python
import random
random.seed(42)

true_win_rates = [0.2, 0.5, 0.8]      # hidden from the agent
estimates = [0.0, 0.0, 0.0]
counts = [0, 0, 0]
epsilon = 0.1                          # 10% of the time, explore

for step in range(2000):
    if random.random() < epsilon:
        arm = random.randrange(3)                      # explore
    else:
        arm = max(range(3), key=lambda a: estimates[a])  # exploit
    reward = 1 if random.random() < true_win_rates[arm] else 0
    counts[arm] += 1
    estimates[arm] += (reward - estimates[arm]) / counts[arm]  # running mean

print("pulls per arm:", counts)
print("best arm learned:", max(range(3), key=lambda a: estimates[a]))
# pulls per arm: [129, 72, 1799]   <- mostly exploits arm 2
# best arm learned: 2
```
The agent was never told which arm is best. It found out from the rewards.

## Common Mistakes / Misconceptions
- **"RL is just supervised learning with rewards."** RL has no correct action labels, and the agent's own choices determine its data.
- **Reward hacking.** Agents exploit flaws in the reward. A boat-racing agent may circle forever collecting points instead of finishing the race. LLMs can learn to flatter a reward model.
- **"RL is only for games and robots."** It is central to aligning and improving reasoning in modern LLMs.
- **Ignoring sample efficiency.** RL often needs huge amounts of interaction.

## Key Takeaways
- RL is learning by trial and error from rewards, not labels.
- Its key concepts are agent, environment, state, action, reward and policy.
- The core trade-off is exploration vs. exploitation; the core difficulty is delayed reward.
- RLHF and RL on verifiable tasks are what turn base LLMs into helpful assistants and reasoners.

## Related Questions
- [What is supervised learning?](./04-What-is-supervised-learning.md)
- [What is unsupervised learning?](./05-What-is-unsupervised-learning.md)
- [How are chat models trained (pretraining, instruction tuning, RLHF)?](../Week-02-Introduction-to-LLMs-and-Generative-AI/15-BONUS-How-are-chat-models-trained-pretraining-instruction-tuning-RLHF.md)
- [What is an AI agent? (LLM vs agent)](../Week-10-AI-Agents-and-Tool-Calling/02-What-is-the-difference-between-an-LLM-and-an-AI-agent.md)
