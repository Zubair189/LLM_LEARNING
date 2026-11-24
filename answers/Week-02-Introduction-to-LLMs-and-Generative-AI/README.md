# Week 02 – Introduction to LLMs and Generative AI

## Overview
This week moves from general AI to the tools you will build with for the rest of the course: large language models (LLMs). You will learn what generative AI is, how an LLM differs from a simple language model and from a scripted chatbot, and what "GPT" stands for. The heart of the week is **tokens**: what they are, how text is cut into them, why word pieces work better than whole words, and how the context window limits what a model can see. You will then link model size (parameters) to memory and cost, tell training apart from inference, see why models hallucinate, and compare open-weight and proprietary models. The bonus question shows how a raw pretrained model becomes a helpful chat assistant. By the end, you should be able to make your first API call, count its tokens, estimate its cost, and explain every part of the request.

## Questions
1. [What is Generative AI?](./01-What-is-Generative-AI.md)
2. [What is a Large Language Model (LLM)?](./02-What-is-a-Large-Language-Model-LLM.md)
3. [How is an LLM different from a traditional chatbot?](./03-How-is-an-LLM-different-from-a-traditional-chatbot.md)
4. [What does GPT stand for?](./04-What-does-GPT-stand-for.md)
5. [What is a language model?](./05-What-is-a-language-model.md)
6. [What is a token?](./06-What-is-a-token.md)
7. [How does tokenization work?](./07-How-does-tokenization-work.md)
8. [Why do LLMs use tokens instead of complete words?](./08-Why-do-LLMs-use-tokens-instead-of-complete-words.md)
9. [What is a context window?](./09-What-is-a-context-window.md)
10. [What are model parameters?](./10-What-are-model-parameters.md)
11. [What is inference?](./11-What-is-inference.md)
12. [Why do LLMs sometimes hallucinate?](./12-Why-do-LLMs-sometimes-hallucinate.md)
13. [What is the difference between an open-weight model and a proprietary model?](./13-What-is-the-difference-between-an-open-weight-model-and-a-proprietary-model.md)
14. [What is the difference between training and inference?](./14-What-is-the-difference-between-training-and-inference.md)
15. **Bonus:** [How are chat models trained (pretraining, instruction tuning, RLHF)?](./15-BONUS-How-are-chat-models-trained-pretraining-instruction-tuning-RLHF.md)

## Practical Exercise (1–2 hours): "Tokens, costs and hallucinations" lab
1. **Build a tokenizer (30 min):** Copy the mini-BPE from [Q7](./07-How-does-tokenization-work.md). Train it on about 1 page of text on a topic you like. Print the first 20 merges, then encode 5 words: 2 common, 2 rare and 1 made-up. Explain how each was split.
2. **Words vs. word pieces vs. letters (15 min):** For the same page, count how many different units there are (vocabulary size) and how many units in a row (sequence length) when you split by words, by letters, and with your BPE (after 50 and 200 merges). Write 3 sentences on what you notice.
3. **First API call and token counting (20 min):** Using the Anthropic SDK (keep `ANTHROPIC_API_KEY` in your environment, never in your code), send one prompt. Print `resp.usage.input_tokens` and `resp.usage.output_tokens`. Compare them with `client.messages.count_tokens(...)` and with the "4 characters per token" rule, and work out the cost using current published prices. *No API key? Run an open-weight model on your own computer with Ollama instead, and estimate tokens with the 4-characters rule.*
4. **Cause and fix a hallucination (25 min):** Ask about something that doesn't exist (e.g. "Summarize the 2019 paper 'Quantum Llamas' by J. Smith"). Save the answer. Then add a system prompt that allows "I don't know" and gives the model a short text to answer from. Compare the two answers.
5. **Write-up (10 min):** A half-page note on what surprised you about tokens and cost, and how giving the model a source text changed its answers.

## Explain Aloud (no notes)
1. **How an LLM turns a prompt into a reply:** cut the text into tokens → read the whole prompt → write the reply one token at a time → turn tokens back into text. Explain where the context window and `max_tokens` limits apply.
2. **Why tokens are word pieces**, and two practical effects (pricing and limits, and weak spelling or arithmetic).
3. **From base model to chat assistant:** pretraining → instruction tuning (SFT) → learning from feedback (RLHF/DPO/RL). Explain why a raw base model doesn't answer questions properly, and why chat models can still hallucinate.
