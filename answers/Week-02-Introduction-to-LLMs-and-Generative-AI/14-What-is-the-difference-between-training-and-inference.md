# What is the difference between training and inference?

> Week 02 – Introduction to LLMs and Generative AI · Question 14

## In One Line
**Training** is when a model **learns** (its numbers change); **inference** is when the finished model is **used** (its numbers stay fixed).

## Simple Explanation
**Training (learning):**
1. The model makes a guess on an example.
2. We measure how wrong it was (the **loss**).
3. We work out which numbers (weights) caused the error. This is called **backpropagation**.
4. We nudge those weights to be a little less wrong. This is called **gradient descent**.
5. Repeat millions or billions of times.

**Inference (using):**
1. Give the model a new input.
2. It runs forward once and gives an answer.
3. Nothing is learned; the weights don't change.

Key differences:

- **How often:** training happens **rarely** (once per model version); inference happens **every time** someone uses the model.
- **Cost:** training a big LLM needs thousands of GPUs for weeks. Inference must be **fast and cheap** for each request.
- **Memory:** training needs much more memory, because it also stores extra numbers used for learning.
- **Who does it:** AI labs train models; most app builders mainly do inference.

Two common mix-ups:
- **Fine-tuning is training**, just smaller, starting from an already-trained model.
- **Giving examples in a prompt is inference.** It helps for that one request, but no weights change.

Question 11 explains how inference works step by step. Week 1's "training vs testing data" is different: it is about splitting data, not about learning vs using.

## Real-Life Example
Training is going to medical school: years of study and correcting mistakes. Inference is the doctor seeing a patient: using what they learned in minutes, without going back to school each time.

## Code Example
The same tiny model `y = w × x`, first trained, then used. The true answer is `w = 3`.

```python
data = [(1, 3), (2, 6), (3, 9), (4, 12)]   # examples: (x, correct y)
w = 0.0                                    # the model's one weight, starts at 0

# ---- TRAINING: guess, measure error, adjust w ----
for epoch in range(200):                   # go over the data 200 times
    for x, y in data:
        guess = w * x
        error = guess - y                  # how wrong?
        w -= 0.01 * 2 * error * x          # nudge w to reduce the error
print("learned w:", round(w, 3))           # learned w: 3.0

# ---- INFERENCE: just use w, never change it ----
def predict(x):
    return w * x
print("prediction for x=10:", round(predict(10), 2))   # prediction for x=10: 30.0
```

**What happened:** during training, `w` changed hundreds of times until it reached 3. During inference, we only **read** `w` to make a prediction.

## Watch Out
- **Chat apps don't learn live from your messages.** Any learning from feedback happens later, in a separate training run.
- **Inference costs add up.** For a popular model, running it for all users can cost more than training it.

## Remember
- Training = learning; weights change.
- Inference = using; weights are frozen.
- Fine-tuning is training; prompting is inference.

## Related Questions
- [What is inference?](./11-What-is-inference.md)
- [What is the difference between training and testing data?](../Week-01-Introduction-to-Artificial-Intelligence/11-What-is-the-difference-between-training-and-testing-data.md)
- [What is backpropagation?](../Week-03-Mathematics-and-Neural-Network-Foundations/14-What-is-backpropagation.md)
- [What is fine-tuning?](../Week-09-Fine-tuning-and-Model-Optimization/02-What-is-fine-tuning.md)
