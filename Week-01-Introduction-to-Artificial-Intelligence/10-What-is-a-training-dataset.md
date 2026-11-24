# What is a training dataset?

> Week 01 – Introduction to Artificial Intelligence · Question 10

## Short Answer
A training dataset is the collection of examples a machine learning model learns from. During training, the model adjusts its parameters to fit the patterns in this data. In supervised learning, each example is an input plus its correct label. In LLM pretraining, it is a huge body of text where the "label" is the next token. The quality, size, diversity and representativeness of the training data largely decide how good, and how biased, the model will be.

## Detailed Explanation
**Anatomy of a (supervised) training dataset:**
| sq_ft | bedrooms | location | **price (label)** |
|---|---|---|---|
| 1200 | 2 | suburb | 250,000 |
| 2500 | 4 | city | 720,000 |
- **Rows** are examples (samples or instances).
- **Feature columns** are the inputs (X).
- **The label column** is the target (y).

**What makes training data good:**
- **Quality:** correct labels, few duplicates, little noise.
- **Quantity:** enough examples for the model's complexity. Bigger models need more data.
- **Diversity / coverage:** includes the edge cases and groups the model will see in production.
- **Representativeness:** matches the real-world distribution. A face detector trained mostly on one demographic will perform worse on others.
- **Legality and ethics:** licensing, consent, and privacy (PII).

**Training data at different scales:**
| Setting | Typical training data |
|---|---|
| Classic ML | Thousands to millions of rows in a CSV or database |
| LLM pretraining | Trillions of tokens: web pages, books, code, papers |
| LLM fine-tuning (SFT) | Thousands to hundreds of thousands of (prompt, response) pairs, often JSONL |
| Preference tuning (RLHF/DPO) | (prompt, preferred answer, rejected answer) triples |

**Where it sits in the data split:** the training set is usually 70–80% of the available labelled data. The rest is held out as **validation** (for tuning) and **test** (for the final unbiased score) data. See Q11.

> **Note on duplication:** Week 9 asks the same question ([What is a training dataset?](../Week-09-Fine-tuning-and-Model-Optimization/05-What-is-a-training-dataset.md)) in the context of **fine-tuning**: chat-formatted JSONL, instruction/response pairs, and data curation for SFT. Treat this Week 1 version as the general ML concept and the Week 9 version as its LLM-specific application.

## Example
**Analogy:** A training dataset is a student's textbook and practice problems. If the textbook has wrong answers (bad labels) or covers only half the syllabus (poor coverage), the student will fail parts of the exam, however clever they are.

**Python: building and splitting a tiny training set:**
```python
import random
random.seed(7)

# sentiment examples: (text, label)
data = [("great product", 1), ("terrible service", 0), ("love it", 1),
        ("waste of money", 0), ("works perfectly", 1), ("broke in a day", 0),
        ("highly recommend", 1), ("very disappointed", 0), ("five stars", 1),
        ("never again", 0)]

random.shuffle(data)                 # avoid ordering bias
split = int(0.8 * len(data))
train, test = data[:split], data[split:]

print(len(train), len(test))         # 8 2
print(sum(y for _, y in train) / len(train))  # 0.5  (fraction of positive labels in train)
```
For an LLM fine-tune, the same idea is usually stored as JSONL, one example per line:
```json
{"messages": [{"role": "user", "content": "Summarize: ..."}, {"role": "assistant", "content": "..."}]}
```

## Common Mistakes / Misconceptions
- **"More data always fixes everything."** Ten thousand clean, relevant examples can beat a million noisy ones.
- **Data leakage.** Test examples, or information from the future, sneak into the training data and inflate the scores.
- **Ignoring class imbalance.** With 99% "not fraud", a model that always says "not fraud" scores 99% accuracy and is useless.
- **Assuming the data is neutral.** Training data encodes historical biases, and models reproduce them.

## Key Takeaways
- The training dataset is what the model learns from; its parameters are fitted to it.
- Quality, coverage and representativeness matter as much as size.
- Always keep separate validation and test sets; never evaluate on the training data.
- LLMs use trillions of tokens for pretraining and much smaller curated sets for fine-tuning.

## Related Questions
- [What is the difference between training and testing data?](./11-What-is-the-difference-between-training-and-testing-data.md)
- [What is supervised learning?](./04-What-is-supervised-learning.md)
- [What is a training dataset? (fine-tuning context)](../Week-09-Fine-tuning-and-Model-Optimization/05-What-is-a-training-dataset.md)
- [What is overfitting?](./12-What-is-overfitting.md)
