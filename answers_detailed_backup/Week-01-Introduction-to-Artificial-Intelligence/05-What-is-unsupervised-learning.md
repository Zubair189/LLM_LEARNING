# What is unsupervised learning?

> Week 01 – Introduction to Artificial Intelligence · Question 5

## Short Answer
Unsupervised learning finds structure in data that has **no labels**. Nobody tells the model the "right answer". Instead it discovers groups, patterns or compact representations by itself. The classic tasks are **clustering** (grouping similar items), **dimensionality reduction** (compressing many features into a few), and **anomaly detection** (spotting unusual points). It is useful because unlabelled data is cheap and plentiful, while labels are expensive.

## Detailed Explanation
| Task | What it does | Common algorithms | Real use |
|---|---|---|---|
| Clustering | Groups similar points | k-means, DBSCAN, hierarchical | Customer segmentation, topic grouping |
| Dimensionality reduction | Compresses features, keeping the structure | PCA, t-SNE, UMAP, autoencoders | Visualizing embeddings, denoising |
| Anomaly detection | Flags points unlike the rest | Isolation Forest, density methods | Fraud, server monitoring |
| Association rules | Finds items that co-occur | Apriori | "People who buy X also buy Y" |

**Supervised vs. unsupervised at a glance:**
```
Supervised:    (x1, "cat") (x2, "dog") ...  → learn x → label
Unsupervised:  x1, x2, x3, ...               → learn "x1 and x3 look alike"
```

**Challenges:**
- **There is no single right answer**, so evaluation is harder. You judge clusters by internal metrics (such as silhouette score) or by whether they are useful for the business.
- **You often must choose hyperparameters** such as *k* (the number of clusters).
- **A human has to interpret the results.** The algorithm says "group 2"; a person decides that group 2 means "bargain hunters".

**Why it matters for LLM engineering:**
- **Embeddings** (Week 7) are learned representations. Clustering or visualizing them is an unsupervised task.
- **LLM pretraining** is *self-supervised*. It uses unlabelled text but builds its own labels (the next token), so it sits between supervised and unsupervised learning. Some textbooks group it with unsupervised learning.

## Example
**Analogy:** You tip a box of mixed Lego bricks onto the floor and sort them into piles without instructions. You might group them by colour or by shape. Nobody told you the categories; you found them yourself.

**Python: k-means clustering from scratch (1-D, standard library):**
```python
import random
random.seed(0)

# customer monthly spend - no labels
spend = [12, 15, 14, 10, 200, 210, 190, 205, 55, 60, 58]

k = 3
centroids = random.sample(spend, k)
for _ in range(10):
    clusters = {i: [] for i in range(k)}
    for x in spend:                                   # assign step
        nearest = min(range(k), key=lambda i: abs(x - centroids[i]))
        clusters[nearest].append(x)
    centroids = [sum(c) / len(c) if c else centroids[i]   # update step
                 for i, c in clusters.items()]

for c in sorted(clusters.values()):
    print(c)
# [12, 15, 14, 10]
# [55, 60, 58]
# [200, 210, 190, 205]
```
The algorithm found "low", "medium" and "high" spenders without being told those groups exist.

## Common Mistakes / Misconceptions
- **"Unsupervised means no human involvement."** Humans still choose features and *k*, and interpret the results.
- **Treating clusters as ground truth.** Clusters depend on the algorithm, the scaling and *k*. Different settings give different groupings.
- **Forgetting to scale features.** With k-means, a feature measured in thousands swamps one measured in decimals.
- **Mixing up self-supervised and unsupervised.** Self-supervised learning builds labels from the data itself (next token, masked word) and trains like supervised learning.

## Key Takeaways
- Unsupervised learning finds structure in unlabelled data.
- Its main tasks are clustering, dimensionality reduction and anomaly detection.
- It is harder to evaluate because there are no correct labels.
- Self-supervised learning, which powers LLM pretraining, is a close cousin of unsupervised learning.

## Related Questions
- [What is supervised learning?](./04-What-is-supervised-learning.md)
- [What is reinforcement learning?](./06-What-is-reinforcement-learning.md)
- [What is a vector?](../Week-03-Mathematics-and-Neural-Network-Foundations/02-What-is-a-vector.md)
- [What is model pretraining?](../Week-09-Fine-tuning-and-Model-Optimization/01-What-is-model-pretraining.md)
