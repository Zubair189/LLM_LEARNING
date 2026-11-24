# How do you evaluate response quality?

> Week 12 – Final Project and Revision · Question 11

> **How this differs from Week 11 Q13 ([What is LLM evaluation?](../Week-11-Production-LLM-Applications/13-What-is-LLM-evaluation.md)):** Q13 is about evaluation as a whole process (test sets, before and after release); this question is the checklist for judging **one single answer**.

## In One Line
You judge each answer against a **checklist**: is it true to the sources, on-topic, complete, in the right format and safe? Code checks the easy parts, an LLM judge checks the rest, and humans check the judge.

## Simple Explanation
"Is this a good answer?" is too vague. Break it into clear questions, a **rubric**:

1. **Faithful:** is every claim backed by the provided documents? (No made-up facts.)
2. **Relevant:** does it answer *this* question, not a different one?
3. **Complete:** does it cover all parts of the question?
4. **Correct:** is it actually right, compared with a known good answer?
5. **Right format:** valid JSON, required fields, citations present, sensible length?
6. **Safe:** no private data leaked, no harmful content, no hidden instructions followed?
7. Bonus: **tone**, and **speed and cost**.

```
Question + documents + answer → Code checks (format, citations, private data)
                              → LLM judge (faithful? relevant? complete? scores 1–5)
                              → Human spot-check (a small sample)
                              → Pass/fail per answer → Dashboard & release gate
```

**Step by step:**
1. **Gather** the question, the documents the model used, and its answer.
2. **Run code checks first.** They are cheap, fast and exact: Is the JSON valid? Is there a citation? Does it contain an email address or card number?
3. **Ask an LLM judge** for the fuzzy parts. Give it a clear rubric (e.g. "5 = every claim is supported, 1 = mostly unsupported"), ask for a reason, and use a low temperature.
4. **Check the judge.** Have humans grade 50–100 answers and see whether the judge agrees. Judges can be biased, for example preferring longer answers.
5. **Combine** the results into a pass or fail with reasons.
6. **Use the scores:** block or flag bad answers live, and run the same test set before every release to catch drops in quality.
7. **Add real feedback:** thumbs up/down, users asking again, and requests for a human.

**For RAG, score retrieval separately.** Was the right chunk even found? If not, fix search. If it was found and the answer is still wrong, fix the prompt or model.

**Example:** documents say "refunds within 30 days". The answer says "refunds within 60 days [refunds.md]". The **format check passes** (it has a citation), but **faithfulness fails** (the number is wrong). That's why you need both kinds of checks.

## Real-Life Example
Grading essays with a marking scheme. A spell-checker catches format problems automatically (code checks). A teaching assistant grades against the scheme (LLM judge). The professor re-grades a few essays to make sure the assistant is fair (human review).

## Watch Out
- Word-overlap scores (like BLEU or ROUGE) can say two answers look similar even when one has the wrong number. Check facts too.
- Don't trust an LLM judge until you've compared it with human grades.

## Remember
- Rubric: **faithful, relevant, complete, correct, right format, safe**.
- **Code checks first**, LLM judge for the fuzzy parts, humans to keep the judge honest.
- In RAG, test **retrieval and generation separately**.

## Related Questions
- [What is LLM evaluation?](../Week-11-Production-LLM-Applications/13-What-is-LLM-evaluation.md)
- [What is hallucination detection?](../Week-11-Production-LLM-Applications/14-What-is-hallucination-detection.md)
- [How can we evaluate RAG answer quality?](../Week-08-Retrieval-Augmented-Generation-RAG/15-How-can-we-evaluate-RAG-answer-quality.md)
- [How do you evaluate a fine-tuned model?](../Week-09-Fine-tuning-and-Model-Optimization/14-How-do-you-evaluate-a-fine-tuned-model.md)
- [What is structured output?](../Week-06-Working-with-LLMs-and-Model-APIs/15-What-is-structured-output.md)
- [How do you deploy an LLM application in production?](12-How-do-you-deploy-an-LLM-application-in-production.md)
