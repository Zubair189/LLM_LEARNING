# Week 06 – Working with LLMs and Model APIs

This week moves from "how LLMs work" to "how to use them in code". You will learn where models come from (pretrained and foundation models, Hugging Face and model repositories), the two main ways to run them (locally with Ollama/`transformers` or in the cloud through an inference API secured by an API key), and how to steer them (system/user prompts, few-shot examples, temperature, top-k/top-p). You will finish with production essentials: streaming responses, getting reliable structured output (and how that differs from JSON mode), and keeping costs and rate limits under control.

## Questions
1. [What is a pretrained model?](01-What-is-a-pretrained-model.md)
2. [What is a foundation model?](02-What-is-a-foundation-model.md)
3. [What is Hugging Face?](03-What-is-Hugging-Face.md)
4. [What is a model repository?](04-What-is-a-model-repository.md)
5. [What is Ollama?](05-What-is-Ollama.md)
6. [What is the difference between local and cloud LLMs?](06-What-is-the-difference-between-local-and-cloud-LLMs.md)
7. [What is an inference API?](07-What-is-an-inference-API.md)
8. [What is an API key?](08-What-is-an-API-key.md)
9. [What is prompt engineering?](09-What-is-prompt-engineering.md)
10. [What is a system prompt?](10-What-is-a-system-prompt.md)
11. [What is a user prompt?](11-What-is-a-user-prompt.md)
12. [What is the temperature parameter?](12-What-is-the-temperature-parameter.md)
13. [What are top-k and top-p sampling?](13-What-are-top-k-and-top-p-sampling.md)
14. [What is response streaming?](14-What-is-response-streaming.md)
15. [What is structured output?](15-What-is-structured-output.md)
16. [What is JSON mode?](16-What-is-JSON-mode.md)
17. [How do you manage API costs and rate limits?](17-How-do-you-manage-API-costs-and-rate-limits.md)
18. **BONUS** – [What is few-shot prompting?](18-BONUS-What-is-few-shot-prompting.md)

Near-duplicate pairs to study together: 1 vs 2 (pretrained = how it was trained; foundation = large general base model), 7 vs 8 (service vs credential), 10 vs 11 (developer rules vs per-turn request), 15 vs 16 (schema-conforming data vs merely valid JSON).

## Practical Exercise (1–2 hours): "Support Ticket Triage – Local vs Cloud"
1. **Setup (15 min):** create a virtual environment; `pip install anthropic pydantic requests python-dotenv`; put `ANTHROPIC_API_KEY` in a `.env` file (git-ignored); install Ollama and `ollama pull llama3.2`.
2. **Data (10 min):** write 10 short fake support tickets in a Python list, each with your expected `category` (billing/shipping/account/bug) and `priority` (1–3).
3. **Schema (10 min):** define a Pydantic `Ticket` model (`category`, `priority`, `summary`).
4. **Cloud classifier (20 min):** use `claude-haiku-5-5` with a system prompt, 3 few-shot examples, and a forced tool call (`tool_choice`) whose `input_schema` is `Ticket.model_json_schema()`. Validate each result.
5. **Local classifier (20 min):** call Ollama `/api/chat` with `"format": Ticket.model_json_schema()` and `temperature: 0`; validate with Pydantic.
6. **Measure (15 min):** for each backend record accuracy vs your labels, validation failures, average latency, and (cloud) total input/output tokens from `resp.usage`. Estimate cost using the current prices from the provider's pricing page.
7. **Robustness (10 min):** wrap cloud calls with handling for `anthropic.RateLimitError` and `anthropic.APIConnectionError`; add a streaming mode that prints a free-text explanation for one ticket.
8. **Write-up:** a short table comparing the two backends, and one paragraph on which you'd choose and why.

## Explain Aloud
Practise explaining each in under 60 seconds, without notes:
1. **Temperature vs top-p** – what each does to the next-token distribution, and which settings you'd use for JSON extraction vs brainstorming.
2. **Structured output vs JSON mode** – why valid JSON isn't enough, and how a forced tool call plus Pydantic validation gives reliable typed data.
3. **Local vs cloud LLMs** – the trade-offs in quality, privacy, cost and operations, and when a hybrid router makes sense.
