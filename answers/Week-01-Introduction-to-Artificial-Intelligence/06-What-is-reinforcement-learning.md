# What is reinforcement learning?

> Week 01 – Introduction to Artificial Intelligence · Question 6

## In One Line
Reinforcement learning is when a computer learns by **trial and error**, getting rewards for good actions and penalties for bad ones.

## Simple Explanation
Nobody tells the computer the right answer. Instead, it tries things and sees what happens.

A few simple words:

- **Agent:** the learner that makes choices (for example, a game-playing program).
- **Environment:** the world it acts in (the game).
- **Action:** a choice the agent makes (a move).
- **Reward:** a score that says "good" or "bad" after an action.
- **Policy:** the strategy the agent learns for picking actions.

The loop goes like this:

1. The agent looks at the situation.
2. It picks an action.
3. It gets a reward (or a penalty).
4. It updates its strategy and repeats, thousands of times.

Two things make this tricky:

- **Delayed rewards:** a good move now may only pay off much later.
- **Explore or exploit?** Should the agent try something new, or repeat what already worked?

**Link to LLMs:** chat models are improved with **RLHF** (Reinforcement Learning from Human Feedback). People rate answers, and the model learns to give the kinds of answers people prefer. Rewards like "did the code pass its tests?" also help train reasoning models.

## Real-Life Example
Training a dog. You don't explain the rules. You give a treat when it sits on command. After many tries, the dog learns which actions earn treats.

## Code Example
Three slot machines with hidden win chances. The agent must find the best one by playing.

```python
import random
random.seed(42)                       # fixed seed so results repeat

win_chance = [0.2, 0.5, 0.8]          # hidden from the agent
total_reward = [0, 0, 0]              # coins won on each machine
tries = [0, 0, 0]                     # times each machine was played

for step in range(2000):
    if random.random() < 0.1 or 0 in tries:
        machine = random.randrange(3)                 # explore: random pick
    else:                                             # exploit: best average so far
        machine = max(range(3), key=lambda m: total_reward[m] / tries[m])
    reward = 1 if random.random() < win_chance[machine] else 0   # play it
    tries[machine] += 1                               # remember the result
    total_reward[machine] += reward

print("tries per machine:", tries)
# Output:
# tries per machine: [63, 73, 1864]
```

**What happened:** the agent was never told which machine was best. From rewards alone, it learned to play machine 2 (the 80% one) almost all the time.

## Watch Out
- **It is not supervised learning.** There are no correct answers given, only rewards.
- **Reward hacking:** an agent may find a silly shortcut that earns points without doing the real task.

## Remember
- RL = learning by **trial and error** with **rewards**.
- Key words: agent, environment, action, reward, policy.
- RLHF uses RL to make chat models more helpful.

## Related Questions
- [What is supervised learning?](./04-What-is-supervised-learning.md)
- [What is unsupervised learning?](./05-What-is-unsupervised-learning.md)
- [How are chat models trained (pretraining, instruction tuning, RLHF)?](../Week-02-Introduction-to-LLMs-and-Generative-AI/15-BONUS-How-are-chat-models-trained-pretraining-instruction-tuning-RLHF.md)
- [What is an AI agent? (LLM vs agent)](../Week-10-AI-Agents-and-Tool-Calling/02-What-is-the-difference-between-an-LLM-and-an-AI-agent.md)
