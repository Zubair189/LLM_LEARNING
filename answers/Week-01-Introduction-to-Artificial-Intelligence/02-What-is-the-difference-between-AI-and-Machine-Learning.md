# What is the difference between AI and Machine Learning?

> Week 01 – Introduction to Artificial Intelligence · Question 2

## In One Line
AI is the **goal** (machines acting smart), and Machine Learning is one **way** to reach it (learning from examples instead of following hand-written rules).

## Simple Explanation
**AI** answers *what* we want: a machine that behaves intelligently.

**Machine Learning (ML)** answers *how* we get it: we show the computer lots of examples, and it finds the patterns by itself. The result is called a **model**, which is just the set of patterns it learned.

So:

1. **Every ML system is AI.** Learning from data is one way to be smart.
2. **Not every AI system is ML.** A chess program built only from rules written by people is AI, but it learns nothing.

**Why did ML become so popular?** Some tasks are far too hard to write rules for. Nobody can write rules that recognise every cat photo. But we can collect millions of cat photos and let the computer learn.

Inside ML there are smaller families: supervised, unsupervised and reinforcement learning. **Deep learning** is also part of ML.

Question 3 covers a similar idea from a programmer's point of view. This question is about **which group sits inside which**.

## Real-Life Example
"Transport" is the goal (AI). A car is one way to travel (ML). Walking and cycling also get you there (rule-based AI). Every car is transport, but not all transport is a car.

## Code Example
The same spam task, solved with a hand-written rule and with learning.

```python
from collections import Counter

emails = ["win money now", "meeting at noon", "win a free prize", "lunch tomorrow?"]
labels = [1, 0, 1, 0]                            # 1 = spam, 0 = normal

def rule_based(text):                            # (a) a HUMAN writes the rule
    return int("win" in text or "free" in text)

spam_words, normal_words = Counter(), Counter()  # (b) the COMPUTER learns
for text, label in zip(emails, labels):          # count words in each kind of email
    if label == 1:
        spam_words.update(text.split())
    else:
        normal_words.update(text.split())

def learned(text):                               # spammy words push the score up
    score = sum(spam_words[w] - normal_words[w] for w in text.split())
    return int(score > 0)                        # positive score = spam

print(rule_based("free money"), learned("free money"))        # 1 1
print(rule_based("lunch at noon"), learned("lunch at noon"))  # 0 0
```

**What happened:** both give the same answers, so both are "AI". Only (b) is machine learning, because its behaviour came from the data. Add more labelled emails and (b) improves with no code changes.

## Watch Out
- **AI and ML are not the same word.** ML is a part of AI.
- **Rule-based systems still count as AI.** They just don't learn.

## Remember
- AI = the goal: smart behaviour.
- ML = one method: learn patterns from data.
- AI ⊃ ML ⊃ Deep Learning.

## Related Questions
- [What is Artificial Intelligence (AI)?](./01-What-is-Artificial-Intelligence-AI.md)
- [How does traditional programming differ from machine learning?](./03-How-does-traditional-programming-differ-from-machine-learning.md)
- [What is supervised learning?](./04-What-is-supervised-learning.md)
- [What is Deep Learning?](./07-What-is-Deep-Learning.md)
