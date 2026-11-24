# Why are Transformers effective at processing language?

> Week 04 – Transformer Architecture · Question 15

## Short Answer
Language is full of context-dependent meaning and long-range relationships: pronouns refer back, verbs agree with distant subjects, and words change meaning with their surroundings. Self-attention lets every token directly consult every other token and build a context-specific representation, so these relationships are easy to model. Multiple heads capture different kinds of relationships in parallel, deep stacks build up from syntax to semantics to reasoning, and the parallel architecture can be trained on enough text to learn all of it.

## Detailed Explanation
**Properties of language and how Transformers handle them**
| Property of language | Transformer mechanism |
|---|---|
| Long-range dependencies ("The **keys** … **were**") | attention connects any two tokens in one step (path length 1) |
| Ambiguity / polysemy ("bank", "bat") | contextual representations: each token's vector is recomputed from its context |
| Many simultaneous relations (syntax, coreference, topic) | multiple attention heads specialise |
| Hierarchical structure (words → phrases → clauses → discourse) | deep stacks of layers build increasingly abstract features |
| Word order matters | positional information (RoPE, sinusoidal) |
| Open vocabulary | sub-word tokens + learned embeddings |
| Huge variety of patterns | parallel training scales to trillions of tokens; scaling laws |

**What layers learn (from interpretability research, simplified)**
- Lower layers: local patterns, parts of speech, previous-token links.
- Middle layers: syntax, entity tracking, coreference.
- Upper layers: semantics, task-relevant features, the next-token decision.
- **Induction heads** copy patterns seen earlier in the context ("A B … A → B"), a mechanism behind in-context learning.

**Flexible, input-dependent processing.** In an RNN or CNN, how information flows between positions is fixed by the architecture. In a Transformer, the attention weights are computed *from the input*, so the network rewires itself for each sentence. That suits language, whose structure varies sentence by sentence.

**Feed-forward layers as memory.** The MLP blocks act roughly like key–value memories storing facts and patterns ("Paris → capital of France"). Attention retrieves the context and the MLP layers supply the knowledge.

**Limits worth knowing**
- $O(n^2)$ attention cost for long documents.
- No built-in guarantee of truthfulness (hallucinations).
- Tokenisation quirks (character counting, arithmetic on digits).
- Fixed context window, so external memory, retrieval (RAG) or long-context techniques are needed for very large inputs.

## Example
**Analogy:** A skilled reader doesn't read strictly word by word with a fading memory (RNN). They glance back and forth across the page, connecting "she" to "Dr. Patel" three sentences earlier while also tracking tone and topic. Each attention head is one of those simultaneous "glances".

**Context example:**
- "She sat on the river **bank**." → attention links "bank" to "river", giving the geography sense.
- "She robbed the **bank**." → attention links "bank" to "robbed", giving the institution sense.

Same input embedding, different contextual output. We can show that mechanism in miniature: the same token vector gets a different output depending on its neighbours.
```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

bank  = np.array([1.0, 1.0, 0.0, 0.0])
river = np.array([2.0, 0.0, 1.0, 0.0])   # "geography" feature in dim 2
rob   = np.array([0.0, 2.0, 0.0, 1.0])   # "crime/finance" feature in dim 3

def contextualise(seq):                   # simple self-attention (Q=K=V=X)
    X = np.stack(seq)
    return softmax(X @ X.T / 2) @ X

print(contextualise([river, bank])[1].round(2))   # [1.5 0.5 0.5 0. ]  bank picks up geography
print(contextualise([rob, bank])[1].round(2))     # [0.5 1.5 0.  0.5]  bank picks up finance
```

## Common Mistakes / Misconceptions
- **"Transformers understand language like humans."** They learn statistical structure extremely well. Whether that equals understanding is debated, and they can still fail on simple-looking tasks.
- **"Attention alone explains their success."** Scale, data quality, FFN memory, residual and normalisation design, and post-training all matter.
- **"They're only good at English."** Multilingual training works well, though quality tracks the amount of data per language.

> **Note on overlap:** [Q3 – Why did Transformers become important?](../Week-04-Transformer-Architecture/03-Why-did-Transformers-become-important.md) focuses on historical and engineering impact (parallelism, scaling). This question focuses on why the *mechanism* fits the structure of language.

## Key Takeaways
- Language needs context, long-range links and multiple simultaneous relationships, which is exactly what self-attention provides.
- Representations are contextual and recomputed for every sentence.
- Depth and multiple heads capture hierarchical and diverse linguistic structure.
- Parallel training makes it possible to learn from enough text to master these patterns.

## Related Questions
- [What is self-attention?](../Week-04-Transformer-Architecture/05-What-is-self-attention.md)
- [What is multi-head attention?](../Week-04-Transformer-Architecture/08-What-is-multi-head-attention.md)
- [Why did Transformers become important?](../Week-04-Transformer-Architecture/03-Why-did-Transformers-become-important.md)
- [What problems do traditional RNNs have?](../Week-04-Transformer-Architecture/01-What-problems-do-traditional-RNNs-have.md)
