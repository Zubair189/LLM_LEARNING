# Explain the complete lifecycle of an LLM.

> Week 12 – Final Project and Revision · Question 1

## In One Line
An LLM is **built** (collect text, learn from it, get trained to be helpful), **tested and released**, then **used** inside apps and **watched and improved** over time.

## Simple Explanation
An LLM's life has seven stages. The first five happen at the AI company. The last two are where app builders like you usually work.

```
Data → Tokenize → Pretrain → Post-train → Test & release → Use in apps → Monitor & improve
                                                                    ↑                 │
                                                                    └──── feedback ───┘
```

**Step by step:**

1. **Collect data.** Gather a huge amount of text: websites, books, code, articles. Clean it by removing duplicates and low-quality or harmful text.
2. **Tokenize.** Split the text into **tokens** (small pieces of words) and turn each into a number, because computers work with numbers.
3. **Pretrain.** A **Transformer** (the model's design, see Q2) reads the text and practises one game over and over: *guess the next token*. Each time it guesses wrong, its internal numbers (**weights**) are adjusted a little. After billions of guesses, it has learned grammar, facts and patterns. This needs thousands of powerful computer chips (GPUs) and lots of money.
4. **Post-train.** The pretrained model can continue text, but it isn't yet a good assistant. So it is trained further on examples of good question-and-answer pairs (**supervised fine-tuning**), and on human preferences for which answers are better and safer.
5. **Test and release.** The company tests it on benchmarks and tries to make it misbehave on purpose (**red-teaming**). Then it is released, either through an API or as downloadable "open weights".
6. **Use in apps (inference).** Your app sends a prompt and the model writes an answer, one token at a time. Here you add prompts, RAG (looking up documents), tools and guardrails.
7. **Monitor and improve.** You watch cost, speed and answer quality. You fix problems with better prompts, better retrieval, fine-tuning, or by switching to a newer model.

**Why this matters to you as an engineer:**
- **Knowledge cutoff:** the model only knows what was in its training data up to a date. For fresh or private facts, use **RAG**.
- **Hallucinations:** the model was trained to write *likely* text, not *checked* truth. So you need grounding and evaluation.
- **Context window:** the amount of text the model can read at once is fixed by its design, so you must manage what you send.
- **Customising options, from cheapest to most expensive:** prompting → RAG → LoRA fine-tuning → full fine-tuning → pretraining from scratch (almost never needed).

Most teams take a ready-made model and own stages 6 and 7.

## Real-Life Example
Training a doctor. **Pretraining** is years of reading every medical textbook. **Post-training** is residency, where seniors correct their behaviour. **Release** is passing the licence exam. **Inference** is seeing patients. **Monitoring** is case reviews and further training. And **RAG** is the doctor checking the patient's file instead of relying on memory.

## Watch Out
- The model does **not** learn from your chats in real time. Its weights are frozen when you use it; only the prompt changes.
- The lifecycle doesn't end at release. Without monitoring, quality can drop without anyone noticing.

## Remember
- **Data → tokenize → pretrain → post-train → test/release → use → monitor.**
- Each stage explains a real behaviour: knowledge cutoff, hallucination, token limits.
- App engineers mostly work on **using** and **monitoring** the model.

## Related Questions
- [What is model pretraining?](../Week-09-Fine-tuning-and-Model-Optimization/01-What-is-model-pretraining.md)
- [What is the difference between training and inference?](../Week-02-Introduction-to-LLMs-and-Generative-AI/14-What-is-the-difference-between-training-and-inference.md)
- [How does tokenization work?](../Week-02-Introduction-to-LLMs-and-Generative-AI/07-How-does-tokenization-work.md)
- [What is supervised fine-tuning (SFT)?](../Week-09-Fine-tuning-and-Model-Optimization/04-What-is-supervised-fine-tuning-SFT.md)
- [What is a foundation model?](../Week-06-Working-with-LLMs-and-Model-APIs/02-What-is-a-foundation-model.md)
- [How do you deploy an LLM application in production?](12-How-do-you-deploy-an-LLM-application-in-production.md)
