# What is the difference between training and testing data?

> Week 01 – Introduction to Artificial Intelligence · Question 11

## Short Answer
**Training data** is what the model learns from: its parameters are adjusted to fit these examples. **Testing data** is a separate held-out set the model never sees during training. It is used once, at the end, to estimate how well the model will perform on new, real-world data. Keeping them separate is essential. A model can memorize its training data, so a score on the training data is overly optimistic. A third set, **validation data**, is used in between to tune hyperparameters and choose between models.

## Detailed Explanation
**The three-way split:**
```
All labelled data
├── Training set   (~70-80%)  → fit model parameters
├── Validation set (~10-15%)  → tune hyperparameters, early stopping, pick model
└── Test set       (~10-15%)  → final, one-time, unbiased performance estimate
```

| | Training | Validation | Test |
|---|---|---|---|
| Used to | Learn the weights | Make design decisions | Report the final score |
| Seen by the model during fitting? | Yes | No (but it influences your choices) | No |
| How often used | Every epoch | Many times | Ideally **once** |

**Why the test set must stay "clean":** every time you look at the test score and change something in response, information leaks from the test set into your design. Eventually you overfit to the test set as well. That is why the validation set exists.

**Reading the gap:**
- Training accuracy 99%, test accuracy 70%: **overfitting** (memorizing).
- Training 65%, test 63%: **underfitting** (the model is too simple).
- Training 92%, test 90%: good generalization.

**Splitting rules of thumb:**
- **Shuffle** before splitting, unless the data is a **time series**. Then split by time (train on the past, test on the future).
- Use a **stratified** split for imbalanced classes, so each split has the same class ratio.
- Avoid **leakage**. Duplicates or near-duplicates across splits, or the same user in both train and test, will inflate the scores.
- For small datasets use **k-fold cross-validation**.

**In the LLM world:** benchmark contamination is a real problem. If benchmark questions appear in web-scale pretraining data, the scores overstate true ability. That is why teams build private, held-out eval sets for their own applications (Week 11).

> **Note on terminology:** don't confuse this with Week 2's [training vs inference](../Week-02-Introduction-to-LLMs-and-Generative-AI/14-What-is-the-difference-between-training-and-inference.md). That question is about the two *phases* of a model's life (learning vs using). This one is about how you *split the data* to measure generalization.

## Example
**Analogy:** A teacher gives students practice problems (training), a mock exam to adjust study plans (validation), and a final exam with **new** questions (test). If the final exam reused the practice problems, a student who memorized the answers would look brilliant without understanding anything.

**Python: a memorizing model looks perfect on training data and fails on test data:**
```python
import random
random.seed(3)

# true rule: label = 1 if x > 50 ; 20% of labels are noise
data = []
for _ in range(200):
    x = random.randint(0, 100)
    y = int(x > 50)
    if random.random() < 0.2:
        y = 1 - y
    data.append((x, y))
train, test = data[:150], data[150:]

lookup = {x: y for x, y in train}          # "memorizer": exact lookup
def memorizer(x):
    return lookup.get(x, 0)

def simple_rule(x):                        # generalizing model
    return int(x > 50)

acc = lambda model, ds: sum(model(x) == y for x, y in ds) / len(ds)
print("memorizer   train/test:", round(acc(memorizer, train), 2), round(acc(memorizer, test), 2))
print("simple rule train/test:", round(acc(simple_rule, train), 2), round(acc(simple_rule, test), 2))
# memorizer   train/test: 0.87 0.62
# simple rule train/test: 0.81 0.78
```
The memorizer has the best training score. (It isn't 100% only because some x values appear twice with conflicting noisy labels.) Its test score, however, collapses to 0.62. The simple rule scores about the same (~0.8) on both. Only the test set reveals which model actually generalizes.

## Common Mistakes / Misconceptions
- **Reporting training accuracy as model quality.** It measures memorization, not generalization.
- **Tuning on the test set.** That turns it into a validation set and leaves you with no honest estimate.
- **Randomly splitting time-series data.** Information from the future leaks into training.
- **Preprocessing before splitting.** Fitting a scaler or vocabulary on all the data leaks test statistics into training.

## Key Takeaways
- Training data teaches; test data checks generalization on unseen examples.
- Validation data is for tuning; the test set is touched once, at the end.
- The train/test gap diagnoses overfitting or underfitting.
- Prevent leakage: split first, de-duplicate, respect time order.

## Related Questions
- [What is a training dataset?](./10-What-is-a-training-dataset.md)
- [What is overfitting?](./12-What-is-overfitting.md)
- [What is underfitting and the bias-variance tradeoff?](./13-BONUS-What-is-underfitting-and-the-bias-variance-tradeoff.md)
- [What is LLM evaluation?](../Week-11-Production-LLM-Applications/13-What-is-LLM-evaluation.md)
