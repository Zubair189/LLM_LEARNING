# What is supervised learning?

> Week 01 – Introduction to Artificial Intelligence · Question 4

## In One Line
Supervised learning is when a computer learns from examples that **come with the correct answer**, so it can guess the answer for new examples.

## Simple Explanation
We give the computer many pairs: an **input** (like a photo) and its correct **label** (like "cat"). A label is simply the right answer attached to an example.

The computer looks at these pairs and learns how inputs connect to answers. Later, it can predict the answer for an input it has never seen.

It is called "supervised" because the labels act like a **teacher** who marks every answer during practice.

There are two main kinds:

1. **Classification:** predict a **category**.
   *Example:* is this email spam or not spam?
2. **Regression:** predict a **number**.
   *Example:* what price will this house sell for?

How it works, step by step:

1. Collect labelled examples.
2. The model makes guesses, and we measure how wrong they are (this measure is called the **loss**).
3. The model adjusts itself to be less wrong, many times over.
4. We check it on **new** examples it never saw, to make sure it really learned.

**Link to LLMs:** fine-tuning a chat model on (question, ideal answer) pairs is supervised learning. LLM pretraining is a close cousin called **self-supervised** learning: the "label" is just the next word in the text.

## Real-Life Example
Think of flashcards. The front shows a picture of a fruit (input), and the back says "apple" (label). After enough cards, you can name fruits you have never seen before.

## Code Example
Guess if a person is a child or an adult by finding the most similar known example.

```python
import math

# (height_cm, weight_kg) -> label   <- every example has an answer
train = [((150, 50), "child"), ((140, 40), "child"),
         ((180, 80), "adult"), ((175, 75), "adult")]

def predict(new_person):
    # find the training example closest to the new person
    closest = min(train, key=lambda ex: math.dist(ex[0], new_person))
    return closest[1]            # copy its label

print(predict((178, 78)))        # adult
print(predict((145, 45)))        # child
```

**What happened:** the program copied the label of the most similar labelled example. This simple method is called **nearest neighbour**.

## Watch Out
- **Don't test on the training examples.** The model will look great there. Always test on new data.
- **Bad labels make a bad model.** If the answers are wrong, the model learns the wrong thing.

## Remember
- Supervised = learning from examples **with answers**.
- Two tasks: **classification** (category) and **regression** (number).
- Success means doing well on **new** data.

## Related Questions
- [What is unsupervised learning?](./05-What-is-unsupervised-learning.md)
- [What is a training dataset?](./10-What-is-a-training-dataset.md)
- [What are training examples and labels?](../Week-09-Fine-tuning-and-Model-Optimization/06-What-are-training-examples-and-labels.md)
- [What is supervised fine-tuning (SFT)?](../Week-09-Fine-tuning-and-Model-Optimization/04-What-is-supervised-fine-tuning-SFT.md)
