# How is RAG different from fine-tuning?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 3

## In One Line
RAG **gives the model information in the prompt** at question time; fine-tuning **changes the model itself** by training it more.

## Simple Explanation
**RAG (Retrieval-Augmented Generation):**
- Searches your documents and puts the relevant pieces into the prompt.
- The model itself (its **weights**, the numbers inside it) never changes.
- To update knowledge, just update the documents. That takes minutes.
- Easy to show sources ("this came from page 12").

**Fine-tuning:**
- Trains the model further on your own examples, so its weights change.
- Best for teaching **behaviour**: a tone of voice, a fixed output format, or a special skill like sorting support tickets.
- To update, you must train again, which takes hours or days and costs money.
- It is a poor way to store facts: the model may remember them wrongly and can't say where they came from.

**A simple rule of thumb:**

1. The model is missing **facts** → use **RAG**.
2. The model answers in the wrong **way** (style, format) → try better prompts first, then **fine-tuning**.
3. You need both → use **both** together.

**Link to LLMs:** most company chatbots use RAG for their knowledge. Fine-tuning is used less often, mainly to shape style or make a smaller model good at one task.

## Real-Life Example
RAG is handing a lawyer the case files before a meeting: they read them and quote them. Fine-tuning is sending the lawyer on a year-long tax course: they now think like a tax expert, but they **still need the case files** for the specific case.

## Watch Out
- Don't fine-tune to "upload documents" into a model. It is unreliable and goes out of date. Use RAG for facts.
- It's not either/or. Many real systems use both.

## Remember
- RAG = knowledge in the prompt; fine-tuning = behaviour in the weights.
- RAG is fast to update and can cite sources.
- Fine-tuning is for style, format and special skills.

## Related Questions
- [What is Retrieval-Augmented Generation (RAG)?](../Week-08-Retrieval-Augmented-Generation-RAG/01-What-is-Retrieval-Augmented-Generation-RAG.md)
- [Why do LLMs need external knowledge?](../Week-08-Retrieval-Augmented-Generation-RAG/02-Why-do-LLMs-need-external-knowledge.md)
- [What is context injection?](../Week-08-Retrieval-Augmented-Generation-RAG/10-What-is-context-injection.md)
