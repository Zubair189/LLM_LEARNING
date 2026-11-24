# What is supervised learning?

> Week 01 – Introduction to Artificial Intelligence · Question 4

## Short Answer
Supervised learning is machine learning from **labelled examples**: pairs of input (X) and correct output (y). The model learns a function that maps inputs to outputs, so it can predict the output for new, unseen inputs. It has two main flavours: **classification**, which predicts a category (spam / not spam), and **regression**, which predicts a number (a house price). It is called "supervised" because the labels act like a teacher that grades each prediction during training.

## Detailed Explanation
**The workflow:**
```
Labelled data (X, y) ──► Train model ──► Evaluate on held-out data ──► Predict on new X
       ▲                    │
       └──── loss = how wrong were the predictions? (drives learning)
```

| Type | Output | Examples | Typical metrics |
|---|---|---|---|
| Classification | Discrete class | Spam detection, sentiment, medical diagnosis | Accuracy, precision, recall, F1 |
| Regression | Continuous number | House price, demand forecast, temperature | MAE, MSE, RMSE |

**Key ingredients:**
- **Features (X)**: the input information, such as square footage or email words.
- **Labels (y)**: the correct answers. Humans often provide them, which is slow and expensive.
- **Loss function**: measures how wrong a prediction is. Training minimizes it.
- **Generalization**: the real goal is good performance on *new* data, not just on the training data.

**Common algorithms:** linear/logistic regression, k-nearest neighbours, decision trees, random forests, gradient boosting, and neural networks.

**Link to LLMs:** LLM pretraining is often called **self-supervised**. The "label" is simply the next word in the text, so no human labelling is needed. Supervised Fine-Tuning (SFT), covered in Week 9, *is* classic supervised learning: (prompt, ideal response) pairs written or curated by people.

## Example
**Analogy:** Learning with flashcards. The front shows a picture of a fruit (input) and the back says "apple" (label). After enough cards you can name fruits you have never seen.

**Python: a 1-nearest-neighbour classifier (pure standard library):**
```python
import math

# (height_cm, weight_kg) -> label
train = [((150, 50), "child"), ((160, 55), "teen"),
         ((180, 80), "adult"), ((175, 75), "adult"), ((140, 40), "child")]

def predict(x):
    # find the training example closest to x, copy its label
    nearest = min(train, key=lambda ex: math.dist(ex[0], x))
    return nearest[1]

print(predict((178, 78)))  # adult
print(predict((145, 45)))  # child
```
The "supervision" is the label attached to every training point.

## Common Mistakes / Misconceptions
- **Evaluating on the training data.** It always looks great there. Use a separate test set.
- **Ignoring label quality.** Noisy or biased labels give a noisy or biased model ("garbage in, garbage out").
- **Mixing up classification and regression.** Predicting a star rating from 1 to 5 can be framed either way, so choose deliberately.
- **Assuming more features are always better.** Irrelevant features add noise and make overfitting more likely.

## Key Takeaways
- Supervised learning learns a mapping X → y from labelled examples.
- Its two tasks are classification (categories) and regression (numbers).
- Labels are the "teacher"; their quality limits the model's quality.
- Success is measured on unseen data (generalization).
- SFT for LLMs is supervised learning; pretraining is self-supervised.

## Related Questions
- [What is unsupervised learning?](./05-What-is-unsupervised-learning.md)
- [What is a training dataset?](./10-What-is-a-training-dataset.md)
- [What are training examples and labels?](../Week-09-Fine-tuning-and-Model-Optimization/06-What-are-training-examples-and-labels.md)
- [What is supervised fine-tuning (SFT)?](../Week-09-Fine-tuning-and-Model-Optimization/04-What-is-supervised-fine-tuning-SFT.md)
