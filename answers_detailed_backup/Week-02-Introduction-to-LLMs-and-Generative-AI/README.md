# Week 02 – Introduction to LLMs and Generative AI

## Overview
This week moves from general AI to the systems you will engineer with for the rest of the course: large language models. You will learn what generative AI is and how an LLM differs from a basic language model and from a scripted chatbot. You will also see where the name "GPT" comes from. The core of the week is the *token economy*: what tokens are, how tokenizers split text, why subwords beat whole words, and how the context window limits what a model can see. You'll then connect model size (parameters) to memory and cost, separate training from inference, understand why models hallucinate, and weigh open-weight against proprietary models. The bonus question explains how a raw pretrained model becomes a helpful chat assistant. By the end you should be able to make a first API call, count its tokens, estimate its cost, and explain every word in the request.

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
1. **Build a tokenizer (30 min):** Copy the mini-BPE from [Q7](./07-How-does-tokenization-work.md). Train it on ~1 page of text from a domain you like. Print the first 20 merges, then encode 5 words: 2 common, 2 rare, and 1 made-up. Explain the splits.
2. **Word vs. subword vs. character (15 min):** For the same page, report vocabulary size and sequence length at the word level, the character level, and with your BPE (after 50 and 200 merges). Write 3 sentences on the trade-off.
3. **First API call + token accounting (20 min):** Using the Anthropic SDK (`ANTHROPIC_API_KEY` in your environment, never in code), send one prompt. Print `resp.usage.input_tokens` and `resp.usage.output_tokens`, compare them with `client.messages.count_tokens(...)` and with the "4 characters per token" estimate, and compute the cost using current published prices. *No API key? Run an open-weight model locally with Ollama instead, and estimate tokens with the heuristic.*
4. **Provoke and fix a hallucination (25 min):** Ask about a fictional entity (e.g. "Summarize the 2019 paper 'Quantum Llamas' by J. Smith"). Record the answer. Then add a system prompt that allows "I don't know" and grounds the model in a short provided context. Compare the two answers.
5. **Write-up (10 min):** A half-page note covering what surprised you about tokens, cost, and how grounding changed the behaviour.

## Explain Aloud (no notes)
1. **How an LLM turns a prompt into a reply:** tokenize → prefill → decode token by token with sampling → detokenize. Explain where the context window and `max_tokens` limits apply.
2. **Why tokens are subwords**, and two practical consequences (pricing/limits, and weak spelling or arithmetic).
3. **From base model to chat assistant:** pretraining → SFT → RLHF/DPO/RL. Explain why a raw base model doesn't answer questions properly and why chat models can still hallucinate.
