# What is fine-tuning?

> Week 09 – Fine-tuning and Model Optimization · Question 2

## In One Line
Fine-tuning means taking a model that is already trained and **training it a little more** on your own small set of examples, so it learns a new behaviour.

## Simple Explanation
A pretrained model already understands language. Fine-tuning does not start from zero. It just **nudges** the model's weights (its internal numbers) using a few hundred or a few thousand examples of what you want.

How it works:

1. Start from an existing model (for example a 7B open model, meaning 7 billion weights).
2. Collect examples: "for this input, reply like this".
3. Train for a short time (1–3 passes over the data) with small, careful steps.
4. Save the result and test it.

There are two main ways to do it:
- **Full fine-tuning:** change *every* weight. Powerful but needs a lot of GPU memory.
- **Parameter-efficient fine-tuning (PEFT)**, such as **LoRA** or **QLoRA:** keep the original weights frozen and train only a small add-on. Cheap enough for one GPU.

**What fine-tuning is good at:** a fixed output format (like JSON), a consistent tone or style, narrow tasks like sorting emails into categories, and making a small model do one job well.

**What it is bad at:** teaching lots of new or changing facts. For that, use **RAG** (looking up documents at question time).

## Real-Life Example
A pretrained model is like a qualified general doctor. Fine-tuning is a short specialist course in skin care. The doctor doesn't relearn medicine from the beginning; they just specialise.

## Code Example
A short LoRA fine-tune using the Hugging Face TRL library (needs a GPU, so it is not run here).

```python
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

ds = load_dataset("json", data_files="train.jsonl", split="train")  # your examples
lora = LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM")       # small add-on to train

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",       # start from an existing model
    train_dataset=ds,
    peft_config=lora,                         # train only the LoRA add-on
    args=SFTConfig(output_dir="out", num_train_epochs=2, learning_rate=2e-4),
)
trainer.train()                    # e.g. training loss falls from ~1.8 to ~0.6
trainer.save_model("out/adapter")  # saves only the small adapter (MBs, not GBs)
```

**What happened:** we loaded our examples, attached a small LoRA add-on, and trained only that add-on. The original model stays unchanged underneath.

## Watch Out
- Fine-tuning to "teach the model our documents" usually works worse than RAG.
- Always compare with a good prompt first. Sometimes a better prompt is enough.

## Remember
- Fine-tuning = a little extra training on your own examples.
- It mainly changes **behaviour and format**, not knowledge.
- LoRA/QLoRA make it affordable on a single GPU.

## Related Questions
- [When should you fine-tune a model?](03-When-should-you-fine-tune-a-model.md)
- [What is full fine-tuning?](11-What-is-full-fine-tuning.md)
- [What is LoRA?](07-What-is-LoRA.md)
- [Can RAG replace fine-tuning?](15-Can-RAG-replace-fine-tuning.md)
