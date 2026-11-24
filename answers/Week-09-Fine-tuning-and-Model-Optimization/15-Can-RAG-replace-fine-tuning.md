# Can RAG replace fine-tuning?

> Week 09 – Fine-tuning and Model Optimization · Question 15

## In One Line
Sometimes, but not always: **RAG gives the model knowledge**, while **fine-tuning changes its behaviour**, so they solve different problems.

## Simple Explanation
**RAG** (Retrieval-Augmented Generation) means: when a question comes in, first search your documents, then paste the most relevant pieces into the prompt so the model can answer from them.

**Fine-tuning** means: train the model on examples so it permanently changes *how* it answers.

**RAG is better when:**
- Facts change often (prices, policies, support tickets). You just update the documents; no retraining.
- You need **sources** ("this answer came from page 4").
- Different users may see different documents. RAG can filter by permission; a fine-tuned model can't "un-know" something for one user.
- You have documents, but not many example question-and-answer pairs.

**Fine-tuning is better when:**
- The problem is *how* to answer, not *what* to know: a strict format, a tone, sorting into categories.
- You want a small, cheap, fast model for a high-volume task.
- Your prompts have become huge with instructions and examples.

**So, can RAG replace it?**
- For "answer questions about our documents": yes, RAG usually replaces fine-tuning completely.
- For "always reply in our exact format and voice, cheaply": no, fine-tuning is the right tool.

**Using both together** is common: fine-tune the model to follow your format and to use the provided documents well (for example, saying "not found" when the documents don't contain the answer), and use RAG to supply the facts.

**Link to LLMs:** agents use the same idea for **long-term memory**: they look up stored notes instead of retraining the model.

## Real-Life Example
A new lawyer joins a firm. Giving them access to the case-file library is RAG: they look things up when needed, and new files appear every day. Training them for months in the firm's writing style is fine-tuning. A good lawyer needs both the library and the training.

## Watch Out
- Fine-tuning to add facts often leads to confident wrong answers and goes out of date quickly.
- RAG fails if the search step finds the wrong documents, so test the search too.

## Remember
- RAG = knowledge; fine-tuning = behaviour.
- For document Q&A, RAG usually wins.
- Many real systems use both.

## Related Questions
- [When should you fine-tune a model?](03-When-should-you-fine-tune-a-model.md)
- [What is fine-tuning?](02-What-is-fine-tuning.md)
- [How do you evaluate a fine-tuned model?](14-How-do-you-evaluate-a-fine-tuned-model.md)
- [What is long-term memory? (Week 10)](../Week-10-AI-Agents-and-Tool-Calling/08-What-is-long-term-memory.md)
