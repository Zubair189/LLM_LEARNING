# How does traditional programming differ from machine learning?

> Week 01 – Introduction to Artificial Intelligence · Question 3

## In One Line
In traditional programming **you write the rules**; in machine learning **you give examples and the computer works out the rules**.

## Simple Explanation
**Traditional programming:** a developer writes exact steps. The computer follows them on the data and gives answers.

> Rules + Data → Answers

**Machine learning:** you give the computer data *and* the correct answers. A training process finds the rules for you. These learned rules are called a **model**. You then use the model on new data.

> Data + Answers → Rules (model)

When to use which:

- **Traditional code** is best when the logic is clear and must be exact, like tax sums or payments.
- **Machine learning** is best when the logic is too messy to write down, like understanding speech or spotting faces.

Some important differences:

1. ML answers can be **wrong sometimes**, so we test them on lots of examples and measure a score.
2. To fix an ML mistake, you usually **fix or add data**, not code.
3. Real apps mix both: lots of normal code around one learned part.

**Link to LLMs:** with LLMs there is now a third style. You "program" a ready-trained model by writing instructions in plain English, called **prompts**.

Question 2 covers a similar topic, but it is about which group sits inside which. This one is about how the work is done.

## Real-Life Example
Traditional programming is giving a new cook a detailed recipe. Machine learning is letting the cook taste 500 dishes, each marked "good" or "bad", until they work out their own recipe.

## Code Example
Converting Celsius to Fahrenheit both ways.

```python
def c_to_f(c):                   # Traditional: WE write the rule
    return c * 9 / 5 + 32
print(c_to_f(25))                # 77.0

# Machine learning: we only give examples (celsius, fahrenheit)
examples = [(0, 32), (10, 50), (20, 68), (30, 86), (40, 104)]
w, b = 0.0, 0.0                  # blank guess for the rule f = w*c + b
for _ in range(20000):           # practise many times
    for c, f in examples:
        error = (w * c + b) - f  # how wrong is the current guess?
        w -= 0.0005 * error * c  # nudge w a little to shrink the error
        b -= 0.0005 * error      # nudge b a little too

print(round(w, 2), round(b, 2))  # 1.8 32.0   <- the rule was learned!
print(round(w * 25 + b, 1))      # 77.0
```

**What happened:** the program was never told the formula. By nudging its guess again and again, it found `w = 1.8` and `b = 32`, which is exactly the real rule.

## Watch Out
- **ML does not replace programming.** Models still need plenty of normal code around them.
- **Don't use ML when a simple rule works.** If the logic is known and must be exact, just write code.

## Remember
- Traditional: rules + data → answers.
- ML: data + answers → rules.
- ML handles messy problems but can make mistakes.

## Related Questions
- [What is the difference between AI and Machine Learning?](./02-What-is-the-difference-between-AI-and-Machine-Learning.md)
- [What is a training dataset?](./10-What-is-a-training-dataset.md)
- [What is the difference between training and testing data?](./11-What-is-the-difference-between-training-and-testing-data.md)
- [What is gradient descent?](../Week-03-Mathematics-and-Neural-Network-Foundations/13-What-is-gradient-descent.md)
