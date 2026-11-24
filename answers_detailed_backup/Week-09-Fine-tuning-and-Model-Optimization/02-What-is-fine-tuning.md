# What is fine-tuning?

> Week 09 – Fine-tuning and Model Optimization · Question 2

## Short Answer
Fine-tuning means continuing to train an already-pretrained model on a smaller, task-specific dataset so that its weights adapt to a new behaviour, style, format or domain. It is much cheaper than pretraining because the model already understands language; you are only nudging it. You can update all of the weights (full fine-tuning) or only a small set of added parameters (parameter-efficient fine-tuning, PEFT, such as LoRA or QLoRA).

## Detailed Explanation
- **Starting point:** a base or instruct checkpoint, e.g. a 7–8B open model.
- **Data:** hundreds to tens of thousands of examples, often chat-format prompt/response pairs.
- **Training:** the same gradient-descent loop as pretraining, but with a small learning rate (e.g. 1e-5 for full fine-tuning, 1e-4 to 2e-4 for LoRA) and only 1–3 epochs.

**Flavours of fine-tuning**

| Type | What changes | Typical use |
|---|---|---|
| Continued pretraining | All weights, raw domain text | Teach domain vocabulary |
| SFT (instruction tuning) | All weights or adapters, on prompt→response pairs | Formats, tasks, tone |
| Preference tuning (DPO/RLHF) | Weights, using chosen-vs-rejected pairs | Helpfulness, style, safety |
| PEFT (LoRA/QLoRA) | Small adapter matrices only | Any of the above, cheaply |

**What fine-tuning is good at:** consistent output format (JSON, a house style), narrow classification or extraction, tone and persona, shorter prompts (behaviour is "baked in"), and distilling a large model's behaviour into a small one.

**What it is bad at:** keeping facts up to date, citing sources, and adding lots of new knowledge. RAG is better for those.

## Example
**Analogy:** a pretrained model is a qualified general doctor. Fine-tuning is a residency in dermatology. They don't relearn medicine; they specialise.

```python
# Minimal LoRA fine-tune with TRL (fits on a single 24 GB GPU for a 7B model)
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

ds = load_dataset("json", data_files="train.jsonl", split="train")
# each line: {"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}

peft_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                      lora_dropout=0.05, task_type="CAUSAL_LM")

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-7B-Instruct",
    train_dataset=ds,
    peft_config=peft_cfg,
    args=SFTConfig(output_dir="out", num_train_epochs=2, learning_rate=2e-4,
                   per_device_train_batch_size=2, gradient_accumulation_steps=8,
                   bf16=True),
)
trainer.train()
trainer.save_model("out/adapter")   # saves only the adapter, tens of MB
```

## Common Mistakes / Misconceptions
- Fine-tuning to "teach the model our docs." Retrieval (RAG) usually works better and stays current.
- Using a pretraining-sized learning rate, which wipes out general skills (catastrophic forgetting).
- Skipping a baseline. Always compare against good prompting or few-shot first.
- Assuming more epochs is better. Small datasets overfit quickly.

## Key Takeaways
- Fine-tuning = further training of a pretrained model on targeted data.
- It mainly changes **behaviour and format**, not broad knowledge.
- PEFT methods (LoRA/QLoRA) make it affordable on one GPU.
- Always evaluate against a prompting/RAG baseline before and after.

## Related Questions
- [When should you fine-tune a model?](03-When-should-you-fine-tune-a-model.md)
- [What is full fine-tuning?](11-What-is-full-fine-tuning.md)
- [What is LoRA?](07-What-is-LoRA.md)
- [Can RAG replace fine-tuning?](15-Can-RAG-replace-fine-tuning.md)
