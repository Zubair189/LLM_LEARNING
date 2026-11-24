# Week 01 – Introduction to Artificial Intelligence

## Overview
This week covers the vocabulary the rest of the course depends on. You will learn what AI is and how machine learning fits inside it. You will see how learning from data differs from writing rules, and meet the three main learning styles: supervised, unsupervised and reinforcement learning. You will learn what neural networks and deep learning are, built up from single neurons, weights and biases. Finally, you will see how data is split and why models overfit or underfit. Each of these ideas comes back in LLM engineering. Pretraining is self-supervised learning, RLHF is reinforcement learning, fine-tuning can overfit, and an LLM is a very large neural network. Aim to explain every concept in plain language and back it up with a tiny code example.

## Questions
1. [What is Artificial Intelligence (AI)?](./01-What-is-Artificial-Intelligence-AI.md)
2. [What is the difference between AI and Machine Learning?](./02-What-is-the-difference-between-AI-and-Machine-Learning.md)
3. [How does traditional programming differ from machine learning?](./03-How-does-traditional-programming-differ-from-machine-learning.md)
4. [What is supervised learning?](./04-What-is-supervised-learning.md)
5. [What is unsupervised learning?](./05-What-is-unsupervised-learning.md)
6. [What is reinforcement learning?](./06-What-is-reinforcement-learning.md)
7. [What is Deep Learning?](./07-What-is-Deep-Learning.md)
8. [What is a neural network?](./08-What-is-a-neural-network.md)
9. [What are neurons, weights and biases?](./09-What-are-neurons-weights-and-biases.md)
10. [What is a training dataset?](./10-What-is-a-training-dataset.md)
11. [What is the difference between training and testing data?](./11-What-is-the-difference-between-training-and-testing-data.md)
12. [What is overfitting?](./12-What-is-overfitting.md)
13. **Bonus:** [What is underfitting and the bias-variance tradeoff?](./13-BONUS-What-is-underfitting-and-the-bias-variance-tradeoff.md)

## Practical Exercise (1–2 hours): "Rules vs. Learning" spam detector
1. **Data (15 min):** Write or collect about 40 short messages (20 spam, 20 not spam) as `(text, label)` pairs in a Python list or CSV. Shuffle them, then split 70/15/15 into train, validation and test sets.
2. **Rule-based baseline (15 min):** Write `is_spam_rules(text)` using hand-picked keywords. Measure its accuracy on the test set.
3. **Learned model (30 min):** Implement a simple word-count classifier, as in Q2, or a Naive Bayes classifier using `collections.Counter`. Train it on the training set only.
4. **Diagnose fit (20 min):** Report training and test accuracy for both approaches. Then make a "memorizer" (an exact-match lookup table) and show its train/test gap. Label each model as underfit, good fit or overfit.
5. **Reflect (10 min):** In 5 sentences, say which approach you would ship, and what you would change if spammers started using new words (a data drift scenario).

*Stretch:* swap in `scikit-learn`'s `CountVectorizer` + `LogisticRegression` and compare.

## Explain Aloud (no notes)
1. **AI ⊃ ML ⊃ Deep Learning ⊃ LLMs.** Draw the nesting and give one example at each level that does *not* belong to the inner level.
2. **What a neuron computes, and how a network learns.** Weighted sum + bias → activation; then loss → backprop → gradient descent.
3. **Overfitting vs. underfitting.** Explain how the train/test gap tells you which one you have, and name two fixes for each.
