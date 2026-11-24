# What is the difference between an open-weight model and a proprietary model?

> Week 02 – Introduction to LLMs and Generative AI · Question 13

## Short Answer
An **open-weight** model publishes its trained parameters (weights) for download, for example Llama, Mistral, Qwen, Gemma or DeepSeek. You can run it on your own hardware, inspect it and fine-tune it, subject to its license. A **proprietary** (closed) model keeps its weights private and is offered only as a hosted service through an API or app, for example Claude, GPT or Gemini. Open weights give you control, privacy and customization, but you run the infrastructure. Proprietary models usually lead on frontier capability and come fully managed, but you depend on the vendor's pricing, policies and availability.

## Detailed Explanation
**"Open-weight" vs. "open-source":** open-weight means the *weights* are released. Fully **open-source** AI, in the stricter sense, also releases the training code, data details, and a permissive license (OSI's Open Source AI Definition). Many popular "open" models are open-weight but have custom licenses with usage restrictions, so always read the license.

| Dimension | Open-weight | Proprietary |
|---|---|---|
| Access | Download the weights (Hugging Face, Ollama) | API / web app only |
| Hosting | You (local GPU, cloud VM, or a third-party host) | Vendor |
| Data privacy | Data can stay entirely on your infrastructure | Sent to the vendor (governed by contract and data-retention terms) |
| Customization | Full fine-tuning, LoRA, quantization, inspecting internals | Prompting; some vendors offer managed fine-tuning |
| Capability | Strong and improving fast; often a step behind the frontier | Usually the frontier (reasoning, long context, tool use) |
| Cost model | Hardware/GPU-hours (cheap at high volume if well utilized) | Pay per token (cheap to start, scales with usage) |
| Ops burden | You handle serving, scaling, updates, security | Managed by the vendor |
| Reproducibility | Pin an exact weight file forever | Model versions are retired over time |
| Safety controls | Whatever you build | Vendor safeguards and usage policies built in |

**How to choose (typical reasoning):**
- **Proprietary** when you need top quality, fast time-to-market, low ops, and moderate volume.
- **Open-weight** when you have strict data residency, offline or edge deployment, very high volume, deep customization, or need fixed versions.
- **Hybrid** is common: a frontier API for hard tasks, and a small local model for cheap classification, embeddings or PII-sensitive steps.

> **Note on overlap:** Week 6's [What is the difference between local and cloud LLMs?](../Week-06-Working-with-LLMs-and-Model-APIs/06-What-is-the-difference-between-local-and-cloud-LLMs.md) is about **where the model runs**. That is a separate axis from **licensing / weight availability**. An open-weight model can run in the cloud (e.g. on a hosted inference provider), and you can't run a proprietary model locally. Keep the two questions distinct: *open vs closed* is about access to weights; *local vs cloud* is about deployment location.

## Example
**Analogy:** A proprietary model is like a restaurant. You order and get excellent food, but you never see the recipe or the kitchen, and the menu and prices can change. An open-weight model is like being handed the finished recipe book. You cook at home, adapt the dishes, and control the ingredients, but you buy the stove and do the dishes. (Open-*source* would also give you the notes on how the recipes were developed.)

**Python: the same task, both ways:**
```python
# Proprietary model via hosted API (Anthropic SDK)
import anthropic
client = anthropic.Anthropic()                 # ANTHROPIC_API_KEY from env
resp = client.messages.create(
    model="claude-sonnet-5-5", max_tokens=100,
    messages=[{"role": "user", "content": "Classify sentiment: 'Loved it!'"}],
)
print(resp.content[0].text)
```
```python
# Open-weight model running locally via Ollama's HTTP API (after `ollama pull llama3.2`)
import json, urllib.request
req = urllib.request.Request(
    "http://localhost:11434/api/chat",
    data=json.dumps({"model": "llama3.2", "stream": False,
                     "messages": [{"role": "user", "content": "Classify sentiment: 'Loved it!'"}]}).encode(),
    headers={"Content-Type": "application/json"},
)
print(json.loads(urllib.request.urlopen(req).read())["message"]["content"])
```
In the second call, your text never leaves your machine.

## Common Mistakes / Misconceptions
- **"Open-weight = open-source = free for any use."** Check the license (commercial limits, user thresholds, acceptable-use terms).
- **"Open-weight models are free."** GPUs, engineering time and ops cost money; at low volume an API is often cheaper.
- **"Proprietary APIs train on my data."** Policies vary by vendor and plan. Many business/API tiers don't train on customer data by default, so read the terms rather than assuming.
- **Mixing up open/closed with local/cloud.** They are independent axes.

## Key Takeaways
- Open-weight: weights downloadable, so you can self-host, fine-tune and inspect; license terms apply.
- Proprietary: weights hidden, accessed through an API; usually frontier quality with zero ops.
- Trade-offs: control/privacy/customization vs. capability/convenience/managed safety.
- Open-weight ≠ open-source; and open/closed is a separate question from local/cloud.

## Related Questions
- [What is the difference between local and cloud LLMs?](../Week-06-Working-with-LLMs-and-Model-APIs/06-What-is-the-difference-between-local-and-cloud-LLMs.md)
- [What is Ollama?](../Week-06-Working-with-LLMs-and-Model-APIs/05-What-is-Ollama.md)
- [What is Hugging Face?](../Week-06-Working-with-LLMs-and-Model-APIs/03-What-is-Hugging-Face.md)
- [What are model parameters?](./10-What-are-model-parameters.md)
