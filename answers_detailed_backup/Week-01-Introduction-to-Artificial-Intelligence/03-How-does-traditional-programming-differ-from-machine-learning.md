# How does traditional programming differ from machine learning?

> Week 01 – Introduction to Artificial Intelligence · Question 3

## Short Answer
In traditional programming, a developer writes explicit rules, and the computer applies them to data to produce answers. In machine learning, the developer supplies data together with the expected answers, and a training algorithm works out the rules (a model). Those learned rules are then applied to new data. Traditional code is best when the logic is clear and must be exact. ML is best when the logic is too complex or fuzzy to write by hand.

## Detailed Explanation
```
Traditional programming:   Rules + Data  ──► [Computer] ──► Answers

Machine learning:          Data + Answers ──► [Training] ──► Rules (model)
                           New data + Model ─────────────► Predictions
```

| Dimension | Traditional programming | Machine learning |
|---|---|---|
| Who writes the logic | Developer | Learned by an algorithm from data |
| Main artifact | Source code | Trained model (weights) + training code |
| Behaviour | Deterministic and exact | Statistical; can be wrong on some inputs |
| Fixing a bug | Edit the code | Fix or add data, change features, retrain |
| Testing | Unit tests with exact expected outputs | Metrics on held-out data (accuracy, F1, ...) |
| Best for | Tax calculation, sorting, payments | Vision, speech, language, recommendations |
| Explainability | High (you can read the code) | Often low (the model is a black box) |

**Engineering consequences:**
- ML systems need **data pipelines**, **evaluation sets**, and **monitoring for drift**. Old code keeps working until someone changes it. A model can get worse on its own when the world changes.
- Real products are **hybrids**. An LLM app is mostly ordinary code (API calls, validation, retries, databases) wrapped around one learned component.
- With LLMs a third style has appeared, sometimes called "Software 3.0". You "program" a pretrained model with natural-language prompts, with no training at all. Weeks 6–10 cover this.

> **Note on overlap:** Q2 ([AI vs ML](./02-What-is-the-difference-between-AI-and-Machine-Learning.md)) is about *taxonomy*. This question is about the *workflow*: who produces the rules, and how you build, test and maintain the system.

## Example
**Analogy:** Traditional programming is giving a new cook a detailed recipe. ML is letting the cook taste 500 finished dishes, each labelled "good" or "bad", until they work out their own recipe.

**Python: converting Celsius to Fahrenheit both ways:**
```python
# Traditional: we know the rule
def c_to_f(c):
    return c * 9 / 5 + 32

# ML: we only have examples (data + answers) and learn w, b in f = w*c + b
data = [(0, 32), (10, 50), (20, 68), (30, 86), (40, 104)]
w, b, lr = 0.0, 0.0, 0.001
for _ in range(20000):                     # gradient descent on squared error
    dw = db = 0.0
    for c, f in data:
        err = (w * c + b) - f
        dw += 2 * err * c / len(data)
        db += 2 * err / len(data)
    w, b = w - lr * dw, b - lr * db

print(c_to_f(25))                 # 77.0
print(round(w, 2), round(b, 2))   # 1.8 32.0   <- the rule was *learned*
print(round(w * 25 + b, 1))       # 77.0
```
The ML version found `w ≈ 1.8` and `b ≈ 32` without ever being told the formula.

## Common Mistakes / Misconceptions
- **"ML replaces programming."** ML systems need a lot of ordinary code around the model: data cleaning, serving, validation and monitoring.
- **Using ML where a simple rule works.** If the logic is known and must be exact (VAT rates, permissions), write code.
- **Expecting ML outputs to be 100% correct.** Plan for errors with confidence thresholds, fallbacks, and human review.
- **Thinking that "more code" fixes an ML bug.** Usually the fix is better data.

## Key Takeaways
- Traditional: rules + data → answers. ML: data + answers → rules.
- ML trades exactness and explainability for the ability to handle messy, complex patterns.
- ML bugs are often data bugs; testing uses metrics on held-out data.
- Production AI systems combine ordinary code with learned components.

## Related Questions
- [What is the difference between AI and Machine Learning?](./02-What-is-the-difference-between-AI-and-Machine-Learning.md)
- [What is a training dataset?](./10-What-is-a-training-dataset.md)
- [What is the difference between training and testing data?](./11-What-is-the-difference-between-training-and-testing-data.md)
- [What is gradient descent?](../Week-03-Mathematics-and-Neural-Network-Foundations/13-What-is-gradient-descent.md)
