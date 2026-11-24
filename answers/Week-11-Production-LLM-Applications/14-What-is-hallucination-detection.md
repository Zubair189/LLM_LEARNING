# What is hallucination detection?

> Week 11 – Production LLM Applications · Question 14

## In One Line
Hallucination detection means **automatically spotting when the model says something false or made up**, so you can block it, fix it or warn the user.

## Simple Explanation
A **hallucination** is when an LLM states something confidently that isn't true or isn't backed by its sources. There are two kinds:
- **Not in the sources:** in RAG, the answer says something the retrieved documents don't say.
- **Simply wrong:** made-up facts, numbers, links or book titles.

**Ways to catch them:**
1. **Ask for citations.** Make the model point to the document it used, e.g. `[refunds.md]`, then check that the document really says that.
2. **Fact-checker model.** A second model (the "judge") compares each claim with the sources and labels it supported or not.
3. **Ask several times.** Generate a few answers. If they disagree with each other, the model is probably guessing.
4. **Check against real data.** Look up numbers, names or dates in a database.

```
Question → Retrieve docs → LLM answer → Check each claim against docs
                                              ├─ supported → send to user
                                              └─ not supported → retry / "I'm not sure" / human
```

**Prevention helps too:** give the model the right documents, tell it to say "I don't know" when unsure, and use a low temperature for factual questions.

## Real-Life Example
A newspaper fact-checker reads a reporter's article and highlights every claim that has no source, before it gets printed.

## Code Example
A very simple checker: for each sentence, what share of its words appear in the source? Real apps use a judge model, but the idea is the same.

```python
context = "The Eiffel Tower is 330 metres tall. It was finished in 1889."
answer = "The Eiffel Tower is 330 metres tall. It was finished in 1889. It has 5 restaurants."

def words(text):
    return set(text.lower().replace(".", "").split())   # simple set of words

context_words = words(context)
for sentence in answer.split(". "):                     # check each claim on its own
    w = words(sentence)
    overlap = len(w & context_words) / len(w)            # share of words found in the source
    label = "supported" if overlap >= 0.8 else "NOT supported - check it!"
    print(f"{overlap:.0%}  {label}  | {sentence.strip('.')}")
# 100%  supported  | The Eiffel Tower is 330 metres tall
# 100%  supported  | It was finished in 1889
# 25%  NOT supported - check it!  | It has 5 restaurants
```

**What happened:** the "5 restaurants" claim had no backing in the source, so it was flagged. Word overlap is crude (it misses rephrasing and negation), which is why real systems use a judge model.

## Watch Out
- RAG **reduces** hallucinations but does not remove them. The model can still misread the documents.
- Detection is never perfect. A fluent, confident answer can still be wrong.

## Remember
- Hallucination = **false or unsupported** content.
- Check claims against sources with citations, judges or real data.
- Decide what to do: **block, retry, warn or send to a human**.

## Related Questions
- [Why do LLMs sometimes hallucinate?](../Week-02-Introduction-to-LLMs-and-Generative-AI/12-Why-do-LLMs-sometimes-hallucinate.md)
- [What is hallucination in RAG?](../Week-08-Retrieval-Augmented-Generation-RAG/12-What-is-hallucination-in-RAG.md)
- [What is LLM evaluation?](13-What-is-LLM-evaluation.md)
- [What are guardrails in LLM applications? (bonus)](19-BONUS-What-are-guardrails-in-LLM-applications.md)
