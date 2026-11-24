# Compare RAG and fine-tuning.

> Week 12 – Final Project and Revision · Question 6

## In One Line
**RAG** gives the model new *knowledge* by putting documents in the prompt; **fine-tuning** changes the model itself to teach it new *behaviour*. In short: RAG for knowledge, fine-tuning for behaviour.

## Simple Explanation
Both are ways to make a general LLM work better for your needs, but they change different things.

```
RAG:         Question → Look up documents → Prompt (rules + documents + question) → Same model → Answer + sources
Fine-tuning: Example pairs (input → ideal output) → Train → New model weights → Question → Answer in the learned style
```

**How RAG works (step by step):**
1. Store your documents in a vector database (chunk and embed them).
2. When a question comes in, find the most relevant chunks.
3. Put those chunks in the prompt and ask the **unchanged** model to answer from them.
4. To update knowledge, just add or replace documents. This takes minutes.

**How fine-tuning works (step by step):**
1. Collect hundreds or thousands of examples: an input and the ideal output.
2. Train the model on them, which changes its **weights** (its internal numbers).
3. Use the new model. It now follows the style, format or skill by habit.
4. To change behaviour again, you must retrain. This takes hours or days.

**When RAG is the better choice:**
- Facts that change often (prices, policies, news).
- Private documents.
- You need **citations** (where did this answer come from?).
- Different users may see different documents (you filter per user).

**When fine-tuning is the better choice:**
- A consistent **tone, style or output format**.
- A narrow specialist task, like sorting tickets into your own categories.
- Making a **smaller, cheaper model** do one job as well as a big one.

**What each costs:**
- RAG is cheaper to start, but each request has a longer prompt.
- Fine-tuning needs good example data, compute and testing up front, but prompts can be shorter later.
- Fine-tuning also needs a model whose weights you can train (open weights or a provider's tuning service).

**A simple decision path:**
1. **Try good prompting first:** clear instructions, a few examples.
2. Missing **knowledge**? Add **RAG**.
3. Behaviour still **inconsistent**? **Fine-tune** (LoRA/QLoRA make it cheaper, see Q7).
4. Need both? Use a fine-tuned model **with** RAG. They work well together.

## Real-Life Example
A new employee. **RAG** is giving them access to the company wiki: they look things up and quote it. **Fine-tuning** is sending them on a training course: they learn *how* to work (writing style, procedures), but they won't know next month's new policy unless they are trained again.

## Watch Out
- Don't fine-tune to add facts that change every week. You'd be retraining forever, and you can't cite sources.
- RAG and fine-tuning aren't rivals. Many real systems use both.

## Remember
- **RAG = knowledge** at question time; **fine-tuning = behaviour** in the weights.
- RAG wins on freshness, citations and per-user access.
- Order to try: **prompting → RAG → fine-tuning → both**.

## Related Questions
- [How is RAG different from fine-tuning?](../Week-08-Retrieval-Augmented-Generation-RAG/03-How-is-RAG-different-from-fine-tuning.md)
- [Can RAG replace fine-tuning?](../Week-09-Fine-tuning-and-Model-Optimization/15-Can-RAG-replace-fine-tuning.md)
- [When should you fine-tune a model?](../Week-09-Fine-tuning-and-Model-Optimization/03-When-should-you-fine-tune-a-model.md)
- [What is prompt engineering?](../Week-06-Working-with-LLMs-and-Model-APIs/09-What-is-prompt-engineering.md)
- [Explain the complete RAG pipeline.](05-Explain-the-complete-RAG-pipeline.md)
- [Explain LoRA and QLoRA.](07-Explain-LoRA-and-QLoRA.md)
