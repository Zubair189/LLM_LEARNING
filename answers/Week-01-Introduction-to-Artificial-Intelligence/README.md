# Week 01 – Introduction to Artificial Intelligence

## Overview
This week teaches the basic words used in the rest of the course. You will learn what AI is and how machine learning fits inside it. You will see how learning from examples differs from writing rules by hand, and meet the three main ways machines learn: supervised, unsupervised and reinforcement learning. You will learn what neural networks and deep learning are, starting from single neurons, weights and biases. Finally, you will see how data is split for testing, and why models can memorise (overfit) or learn too little (underfit). All of these ideas come back later with LLMs. For example, an LLM is a very large neural network, and its chat skills are improved with reinforcement learning. Try to explain each idea in plain words and back it up with a tiny code example.

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
1. **Data (15 min):** Write about 40 short messages (20 spam, 20 not spam) as `(text, label)` pairs in a Python list or CSV. Shuffle them, then split them 70/15/15 into training, validation and test sets.
2. **Rules first (15 min):** Write `is_spam_rules(text)` using keywords you pick by hand. Check how many test messages it gets right (its accuracy).
3. **Learned model (30 min):** Build a simple word-counting spam detector, like the one in Q2 (or a Naive Bayes one using `collections.Counter`). Train it on the training set only.
4. **Check the fit (20 min):** Print training and test accuracy for both approaches. Then build a "memorizer" (a lookup table of exact messages) and show the gap between its training and test scores. Label each model as underfit, good fit or overfit.
5. **Reflect (10 min):** In 5 sentences, say which approach you would use for real, and what you would change if spammers started using new words (the data "drifting" over time).

*Extra challenge:* try `scikit-learn`'s `CountVectorizer` + `LogisticRegression` and compare.

## Explain Aloud (no notes)
1. **AI ⊃ ML ⊃ Deep Learning ⊃ LLMs.** Draw the boxes inside boxes, and give one example at each level that does *not* belong to the next box in.
2. **What a neuron does, and how a network learns.** Weighted sum + bias → activation. Then: measure the error (loss) → find which weights caused it (backpropagation) → nudge them (gradient descent).
3. **Overfitting vs. underfitting.** Explain how comparing training and test scores tells you which one you have, and name two fixes for each.
