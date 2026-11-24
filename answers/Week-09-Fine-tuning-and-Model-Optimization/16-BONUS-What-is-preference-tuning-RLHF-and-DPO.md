# What is preference tuning (RLHF and DPO)?

> Week 09 – Fine-tuning and Model Optimization · Bonus question (added) · BONUS

## In One Line
Preference tuning teaches a model by showing it **two answers and which one people liked better**, so it learns to give the better kind of answer.

## Simple Explanation
With SFT, every example has one "correct" answer. But for many questions there's no single right answer, only better and worse ones. People find it easy to say "answer A is better than B", even when they can't write the perfect answer themselves.

So preference data looks like this: a prompt, a **chosen** (better) answer, and a **rejected** (worse) answer.

It usually happens **after SFT**:

1. Take the SFT model.
2. Let it write two or more answers to each prompt.
3. Humans (or an AI judge) pick which answer is better.
4. Train the model to make the chosen kind of answer more likely and the rejected kind less likely.

There are two famous ways to do step 4:

- **RLHF** (Reinforcement Learning from Human Feedback): first train a separate **reward model** that learns to score answers like a human would. Then let the LLM practise and improve its score using reinforcement learning (classically an algorithm called PPO). Powerful, but complicated and finicky.
- **DPO** (Direct Preference Optimization): skip the reward model. Train the LLM directly on the chosen/rejected pairs with a simple formula. It's much simpler and cheaper, and it works with LoRA.

Both keep the model close to its starting version, so it doesn't drift into strange behaviour. In DPO, a setting called `beta` (often around 0.1) controls how strongly that holds.

**Link to LLMs:** this step is a big reason chat models feel helpful, polite and safe.

## Real-Life Example
Learning to cook with a friend who tastes two versions of your soup and says "this one's better". You never get a perfect recipe, but after many comparisons your soup steadily improves.

## Code Example
A toy version of the DPO idea: the loss is small when the model prefers the chosen answer.

```python
import math

def dpo_loss(chosen_gain, rejected_gain, beta=0.1):
    # "gain" = how much MORE likely the model now finds an answer than the original model did
    margin = beta * (chosen_gain - rejected_gain)     # how strongly it prefers "chosen"
    return round(-math.log(1 / (1 + math.exp(-margin))), 3)

print(dpo_loss(0, 0))     # 0.693  <- no preference yet
print(dpo_loss(5, -5))    # 0.313  <- prefers the chosen answer: lower loss
print(dpo_loss(-5, 5))    # 1.313  <- prefers the rejected answer: higher loss
```

**What happened:** training pushes this loss down, which means making chosen answers more likely and rejected ones less likely, compared with the original model.

## Watch Out
- Preference tuning doesn't add knowledge; it shapes style, helpfulness and safety.
- Bad or inconsistent preference labels teach bad habits, just like bad SFT data.

## Remember
- Data = prompt + chosen answer + rejected answer.
- RLHF uses a reward model + reinforcement learning; DPO trains directly and is simpler.
- It usually comes right after SFT.

## Related Questions
- [What is supervised fine-tuning (SFT)?](04-What-is-supervised-fine-tuning-SFT.md)
- [What are training examples and labels?](06-What-are-training-examples-and-labels.md)
- [How do you evaluate a fine-tuned model?](14-How-do-you-evaluate-a-fine-tuned-model.md)
- [What is model pretraining?](01-What-is-model-pretraining.md)
