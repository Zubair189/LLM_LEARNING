# Why do LLMs sometimes hallucinate?

> Week 02 – Introduction to LLMs and Generative AI · Question 12

## In One Line
LLMs hallucinate (say false things confidently) because they are trained to produce text that **sounds likely**, not to **check facts**.

## Simple Explanation
A **hallucination** is when a model gives an answer that sounds fluent and confident but is wrong or made up. Examples: a fake book title, a wrong date, or a link that doesn't exist.

Why does it happen?

1. **It predicts likely words, not true facts.** When it doesn't know something, it still produces something that *sounds* right, instead of saying "I don't know".
2. **Gaps in knowledge.** It knows nothing after its training cutoff, and nothing about your private data.
3. **Blurry memory.** Billions of facts are squeezed into its weights, so rare facts get mixed up.
4. **Training rewarded confident answers.** Guessing often scored better than admitting uncertainty.
5. **Tricky prompts.** Questions with a false premise, like "Why did Einstein win the 1950 Nobel Prize?" (he won in 1921), can lead it astray.

How to reduce it:

- **Give it the facts.** Put the source text in the prompt, or use **RAG**: automatically fetch relevant documents and add them to the prompt.
- **Allow "I don't know"** in your instructions.
- **Ask for quotes** from the source, then check them with code.
- **Check important answers**, with a second check or a human.

Hallucinations can be **reduced a lot**, but not removed completely.

Weeks 8 and 11 cover hallucinations in RAG and how to detect them. This question explains why they happen at all.

## Real-Life Example
Think of a confident student in an oral exam who doesn't know the answer but keeps talking in the right style. It sounds convincing, but the details are invented.

## Code Example
A simple check: is the model's quoted claim really in the source text?

```python
source = "Acme's refund window is 30 days. Refunds go to the original payment method."

claims = ["refund window is 30 days",     # what a model might quote back
          "refund window is 60 days"]     # a hallucinated detail

for claim in claims:
    print(claim, "->", claim in source)   # True only if the exact words are in the source

# Output:
# refund window is 30 days -> True
# refund window is 60 days -> False
```

**What happened:** the true claim was found in the source, while the made-up one was caught. Real systems combine checks like this with instructions such as "Answer only from the context, or say I don't know."

## Watch Out
- **RAG doesn't fix everything.** A model can still misread or ignore the documents you give it.
- **Temperature 0 doesn't stop hallucinations.** Always picking the most likely word can still be confidently wrong.

## Remember
- LLMs aim for **likely** text, not **true** text.
- Grounding, "I don't know", and checking reduce hallucinations.
- Always verify important answers.

## Related Questions
- [What is hallucination in RAG?](../Week-08-Retrieval-Augmented-Generation-RAG/12-What-is-hallucination-in-RAG.md)
- [What is hallucination detection?](../Week-11-Production-LLM-Applications/14-What-is-hallucination-detection.md)
- [Why do LLMs need external knowledge?](../Week-08-Retrieval-Augmented-Generation-RAG/02-Why-do-LLMs-need-external-knowledge.md)
- [What is a language model?](./05-What-is-a-language-model.md)
