# What is the difference between AI and Machine Learning?

> Week 01 – Introduction to Artificial Intelligence · Question 2

## Short Answer
AI is the broad goal of making machines behave intelligently. Machine Learning (ML) is one way to reach that goal: the machine learns patterns from data instead of following rules a human wrote. Every ML system is AI, but not every AI system uses ML. A hand-coded chess engine is AI without ML, and a spam filter trained on labelled emails is AI built with ML.

## Detailed Explanation
- **AI** answers *what* we want: intelligent behaviour.
- **ML** answers *how* we get it: by fitting a model to data.

| Aspect | AI (broad) | Machine Learning (subset) |
|---|---|---|
| Scope | Any technique that produces intelligent behaviour | Algorithms that improve with data/experience |
| Knowledge source | Hand-written rules, search, logic, *or* learned | Learned from examples |
| Example techniques | Expert systems, A* search, planning, ML | Linear regression, decision trees, neural networks |
| Behaviour change | Only if a human edits the rules (for rule-based AI) | Retrain on new data |

Tom Mitchell's classic definition of ML: *a program learns from experience E with respect to task T and performance measure P if its performance at T, measured by P, improves with E.*

**Where ML sits:**
```
AI
 ├── Rule-based / symbolic (search, logic, expert systems)
 └── Machine Learning
       ├── Supervised      (labelled data)
       ├── Unsupervised    (unlabelled data)
       ├── Reinforcement   (rewards)
       └── Deep Learning   (multi-layer neural nets; applies to all three above)
```

**Why ML took over:** Many tasks are too hard to write rules for. Nobody can write down rules that recognize every cat photo or understand every sentence, but you can collect millions of examples.

> **Note on overlap:** Q3 ([traditional programming vs ML](./03-How-does-traditional-programming-differ-from-machine-learning.md)) covers similar ground from a software-engineering angle: *rules + data → answers* versus *data + answers → rules*. This question is about the **taxonomy** (what is a subset of what).

## Example
**Analogy:** "Transport" is the goal (AI). A car is one way to get there (ML). Walking and cycling also get you there (rule-based AI, search). All cars are transport, but not all transport is a car.

**Python: the same task done as rule-based AI and as ML:**
```python
# Task: classify a message as spam (1) or not (0)
emails = ["win money now", "meeting at noon", "win a free prize", "lunch tomorrow?"]
labels = [1, 0, 1, 0]

# (a) Rule-based AI: a human writes the rule
def rule_based(text):
    return int("win" in text or "free" in text)

# (b) Machine learning: learn word "spamminess" from labelled data
from collections import Counter
spam_words, ham_words = Counter(), Counter()
for text, y in zip(emails, labels):
    (spam_words if y else ham_words).update(text.split())

def learned(text):
    score = sum(spam_words[w] - ham_words[w] for w in text.split())
    return int(score > 0)

test = "free money"
print(rule_based(test), learned(test))  # 1 1
```
Both are "AI". Only (b) is machine learning, because its behaviour came from the data. Add new labelled emails and (b) changes with no code edits.

## Common Mistakes / Misconceptions
- **Using "AI" and "ML" as synonyms.** In interviews, show that you know ML is a subset.
- **"Rule-based means not AI."** Expert systems and game-tree search are textbook AI.
- **"ML models understand rules."** They learn statistical associations, which may or may not match the real causal rules.
- **"Deep learning is separate from ML."** It is a subset of ML.

## Key Takeaways
- AI is the goal (intelligent behaviour). ML is a data-driven method for reaching it.
- ML ⊂ AI. Deep learning ⊂ ML.
- ML is preferred when rules are too complex to write by hand but examples are plentiful.
- ML systems improve by retraining on data, not by a human editing code.

## Related Questions
- [What is Artificial Intelligence (AI)?](./01-What-is-Artificial-Intelligence-AI.md)
- [How does traditional programming differ from machine learning?](./03-How-does-traditional-programming-differ-from-machine-learning.md)
- [What is supervised learning?](./04-What-is-supervised-learning.md)
- [What is Deep Learning?](./07-What-is-Deep-Learning.md)
