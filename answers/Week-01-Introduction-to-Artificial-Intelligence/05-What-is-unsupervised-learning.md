# What is unsupervised learning?

> Week 01 – Introduction to Artificial Intelligence · Question 5

## In One Line
Unsupervised learning is when a computer looks at data **without any answers given** and finds patterns or groups by itself.

## Simple Explanation
In **supervised learning**, we give the computer examples *with* the correct answer, like photos labelled "cat" or "dog".

In **unsupervised learning**, we give it only the data and **no labels**. Nobody tells it what is right. The computer's job is to notice on its own which things look similar and which look different.

It mainly does three things:

1. **Grouping (clustering):** puts similar items together.
   *Example:* splitting customers into "low spenders", "medium spenders" and "big spenders".
2. **Simplifying (dimensionality reduction):** squeezes data with many details into a few important ones, so it is easier to see and work with.
3. **Spotting odd ones out (anomaly detection):** finds items that don't fit the usual pattern.
   *Example:* a strange bank transaction that might be fraud.

**Why is this useful?** Labelling data takes people a lot of time and money. Unlabelled data, like raw text, photos or logs, is everywhere and free. Unsupervised learning lets us use it.

**Link to LLMs:** LLMs learn from huge amounts of unlabelled text. They make their own "answer" by trying to guess the next word. This is called **self-supervised learning**, a close cousin of unsupervised learning.

## Real-Life Example
Imagine tipping a big box of mixed Lego bricks onto the floor. Nobody gives you instructions, but you naturally sort them into piles: red ones here, blue ones there, small ones in another pile. You **found the groups yourself**. That is unsupervised learning.

## Code Example
Group customers by how much they spend each month. We never tell the program what the groups are.

```python
# Monthly spend of 11 customers (no labels!)
spend = [12, 15, 14, 10, 200, 210, 190, 205, 55, 60, 58]

# Start with 3 guesses for the "centre" of each group
centres = [10, 60, 200]

for _ in range(5):                      # repeat a few times to improve
    groups = [[], [], []]
    for x in spend:
        # put each customer in the group with the closest centre
        closest = min(range(3), key=lambda i: abs(x - centres[i]))
        groups[closest].append(x)
    # move each centre to the average of its group
    centres = [sum(g) / len(g) for g in groups]

for g in groups:
    print(g)

# Output:
# [12, 15, 14, 10]        <- low spenders
# [55, 60, 58]            <- medium spenders
# [200, 210, 190, 205]    <- big spenders
```

**What happened:** the program found three natural groups just by looking at which numbers are close to each other. This method is called **k-means clustering**.

## Watch Out
- The computer only says "group 1, group 2…". **A human still decides** what each group means.
- There is no single "correct" answer, so it is harder to check whether the result is good.

## Remember
- Unsupervised learning = **data without labels**.
- The computer finds **groups and patterns** by itself.
- Common uses: customer grouping, fraud detection, organising documents.

## Related Questions
- [What is supervised learning?](./04-What-is-supervised-learning.md)
- [What is reinforcement learning?](./06-What-is-reinforcement-learning.md)
- [What is model pretraining?](../Week-09-Fine-tuning-and-Model-Optimization/01-What-is-model-pretraining.md)
