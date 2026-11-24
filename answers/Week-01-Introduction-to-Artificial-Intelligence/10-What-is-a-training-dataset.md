# What is a training dataset?

> Week 01 – Introduction to Artificial Intelligence · Question 10

## In One Line
A training dataset is the **collection of examples a model learns from**.

## Simple Explanation
During training, the model looks at these examples again and again and adjusts its numbers to fit the patterns in them.

What one example looks like depends on the task:

- **Supervised learning:** an input plus its correct answer (label).
  *Example:* house size and location → price.
- **LLM pretraining:** huge amounts of plain text. The "answer" is simply the next word.
- **LLM fine-tuning:** pairs of (question, ideal reply).

What makes training data **good**:

1. **Correct:** the labels are right.
2. **Enough:** there are enough examples for the job.
3. **Varied:** it covers the different cases the model will meet in real life.
4. **Fair and legal:** it is not one-sided, and you are allowed to use it.

We never use *all* our data for training. We keep some aside (usually 20–30%) to **test** the model later on examples it has never seen. See Question 11.

**Link to LLMs:** pretraining uses **trillions** of words from the web, books and code. Fine-tuning uses much smaller, carefully chosen sets.

Week 9 asks the same question for fine-tuning. This Week 1 version is the general idea.

## Real-Life Example
A training dataset is like a student's textbook and practice problems. If the book has wrong answers, or covers only half the subject, the student will struggle in the exam, however clever they are.

## Code Example
Build a tiny training set of reviews and keep some aside for testing.

```python
import random
random.seed(7)                        # fixed seed so results repeat

# each example: (text, label)   1 = positive, 0 = negative
data = [("great product", 1), ("terrible service", 0), ("love it", 1),
        ("waste of money", 0), ("works perfectly", 1), ("broke in a day", 0),
        ("highly recommend", 1), ("very disappointed", 0), ("five stars", 1),
        ("never again", 0)]

random.shuffle(data)                  # mix them up first
split = int(0.8 * len(data))          # 80% for training
train, test = data[:split], data[split:]

print(len(train), len(test))          # 8 2
print(train[0])                       # ('five stars', 1)
```

**What happened:** we shuffled the examples and used 8 of them to train, keeping 2 hidden for testing later.

## Watch Out
- **More data is not always better.** A small clean set can beat a huge messy one.
- **Data is not neutral.** If the data is biased, the model will copy that bias.

## Remember
- Training data = the examples the model learns from.
- Quality and variety matter as much as size.
- Always keep some data aside for testing.

## Related Questions
- [What is the difference between training and testing data?](./11-What-is-the-difference-between-training-and-testing-data.md)
- [What is supervised learning?](./04-What-is-supervised-learning.md)
- [What is a training dataset? (fine-tuning context)](../Week-09-Fine-tuning-and-Model-Optimization/05-What-is-a-training-dataset.md)
- [What is overfitting?](./12-What-is-overfitting.md)
