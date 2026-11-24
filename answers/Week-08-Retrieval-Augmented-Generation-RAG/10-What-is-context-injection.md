# What is context injection?

> Week 08 – Retrieval-Augmented Generation (RAG) · Question 10

## In One Line
Context injection means **putting the retrieved chunks into the LLM's prompt**, with clear instructions on how to use them.

## Simple Explanation
After retrieval finds the best chunks, we must hand them to the LLM. We do this by writing them into the prompt. This is the "Augmented" step in RAG. The chunks are called the **context**.

A good prompt has three clear parts:

1. **Instructions** (usually in the system prompt): "Answer only from the context. Cite sources like [1]. If the answer isn't there, say you don't know."
2. **Context:** the chunks, each wrapped in a label with its source, e.g. `<doc id="1" source="handbook p.12">…</doc>`.
3. **The question** from the user.

**Good habits:**
- **Label each chunk** so the model can cite it and so it's clear where evidence starts and ends.
- **Put the best chunks first.** Models pay most attention to the start and end of long prompts.
- **Don't add too much.** Extra text costs money and can distract the model.
- **Always include the "say you don't know" rule.** Otherwise the model may fill gaps by guessing.
- **Treat chunks as data, not commands.** A document might contain "ignore your instructions"; this trick is called **prompt injection**. Tell the model to only use the context as information.

**Link to LLMs:** how you arrange the context in the prompt has a big effect on answer quality.

## Real-Life Example
Briefing a spokesperson before a press conference. You hand them a labelled folder of approved facts (context) and say "only use these, mention which document, and say 'no comment' if it's not in the folder" (instructions). Then the reporter asks a question.

## Code Example
Building the prompt (this part is plain Python):

```python
chunks = [   # what the retriever found
    {"text": "Full-time staff get 25 vacation days.", "source": "handbook p.12"},
    {"text": "Part-time staff get vacation pro rata.", "source": "handbook p.13"},
]
question = "How many vacation days do full-time staff get?"

# Wrap each chunk in a labelled tag so the model can cite it
context = "\n".join(
    f'<doc id="{i}" source="{c["source"]}">{c["text"]}</doc>'
    for i, c in enumerate(chunks, start=1)
)
prompt = f"<context>\n{context}\n</context>\n\nQuestion: {question}"
print(prompt)
# <context>
# <doc id="1" source="handbook p.12">Full-time staff get 25 vacation days.</doc>
# <doc id="2" source="handbook p.13">Part-time staff get vacation pro rata.</doc>
# </context>
#
# Question: How many vacation days do full-time staff get?
```

**What happened:** we built a tidy prompt with labelled evidence followed by the question. Next you would send it with `client.messages.create(model="claude-sonnet-5-5", max_tokens=1024, system="Answer only from the context...", messages=[{"role": "user", "content": prompt}])`.

## Watch Out
- Dumping raw chunks with no labels makes citations impossible and confuses the model.
- Without a "say you don't know" rule, the model may answer from memory and invent things.

## Remember
- Context injection = putting retrieved chunks + instructions into the prompt.
- Label chunks, put the best first, keep it short.
- Treat retrieved text as information, not instructions.

## Related Questions
- [What is a RAG pipeline?](../Week-08-Retrieval-Augmented-Generation-RAG/11-What-is-a-RAG-pipeline.md)
- [What is hallucination in RAG?](../Week-08-Retrieval-Augmented-Generation-RAG/12-What-is-hallucination-in-RAG.md)
- [What is the difference between retrieval and generation?](../Week-08-Retrieval-Augmented-Generation-RAG/14-What-is-the-difference-between-retrieval-and-generation.md)
