# What is the difference between training and testing data?

> Week 01 – Introduction to Artificial Intelligence · Question 11

## In One Line
**Training data** is what the model learns from; **testing data** is kept hidden and used at the end to check how well it works on new examples.

## Simple Explanation
A model can **memorise** its training examples. So a good score on training data does not prove much. To know if it really learned, we test it on examples it has **never seen**.

We usually split our data into three parts:

1. **Training set (about 70–80%):** the model learns from this.
2. **Validation set (about 10–15%):** we use this while building, to try settings and pick the best version.
3. **Test set (about 10–15%):** used **once**, at the very end, for an honest final score.

Comparing the scores tells you a lot:

- Training high, test much lower → the model **memorised** (overfitting).
- Both low → the model is **too simple** (underfitting).
- Both high and close → it **learned well**.

Simple rules:
- Shuffle before splitting, but for data over time, train on the past and test on the future.
- Never let test examples leak into training.

**Link to LLMs:** if test questions accidentally appear in a model's training text, its scores look better than they really are.

Don't confuse this with "training vs inference" (Week 2). That is about learning vs using a model. This is about **splitting the data**.

## Real-Life Example
A teacher gives practice problems (training), a mock exam (validation) and a final exam with **new** questions (test). If the final reused the practice questions, a student who memorised answers would look brilliant without understanding anything.

## Code Example
A "memoriser" vs. a simple rule. The true pattern is "label 1 if x > 50", with some random noise.

```python
import random
random.seed(3)

data = []
for _ in range(200):
    x = random.randint(0, 100)
    y = int(x > 50)                       # true rule
    if random.random() < 0.2:             # 20% of labels are noisy (flipped)
        y = 1 - y
    data.append((x, y))
train, test = data[:150], data[150:]      # split: 150 train, 50 test

lookup = {x: y for x, y in train}         # memoriser: remember every answer
def memorizer(x):
    return lookup.get(x, 0)               # unknown x? just guess 0
def simple_rule(x):
    return int(x > 50)

def accuracy(model, rows):                # fraction of correct answers
    return round(sum(model(x) == y for x, y in rows) / len(rows), 2)

print("memorizer   train:", accuracy(memorizer, train), "test:", accuracy(memorizer, test))
print("simple rule train:", accuracy(simple_rule, train), "test:", accuracy(simple_rule, test))
# memorizer   train: 0.87 test: 0.62
# simple rule train: 0.81 test: 0.78
```

**What happened:** the memoriser looked best on training data but dropped badly on the test set. Only the test set showed that the simple rule really works.

## Watch Out
- **Don't keep tweaking based on the test score.** Then it stops being an honest test. Use the validation set for that.
- **Training accuracy is not model quality.** It mostly shows how well the model remembers.

## Remember
- Training data teaches; test data checks.
- Use the test set **once**, at the end.
- A big gap between training and test scores means memorising.

## Related Questions
- [What is a training dataset?](./10-What-is-a-training-dataset.md)
- [What is overfitting?](./12-What-is-overfitting.md)
- [What is underfitting and the bias-variance tradeoff?](./13-BONUS-What-is-underfitting-and-the-bias-variance-tradeoff.md)
- [What is LLM evaluation?](../Week-11-Production-LLM-Applications/13-What-is-LLM-evaluation.md)
