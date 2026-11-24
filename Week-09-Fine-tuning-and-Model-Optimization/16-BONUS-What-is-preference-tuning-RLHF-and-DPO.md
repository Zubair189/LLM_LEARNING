# What is preference tuning (RLHF and DPO)?

> Week 09 – Fine-tuning and Model Optimization · Bonus question (added) · BONUS

## Short Answer
Preference tuning trains a model on **comparisons**, of the form "for this prompt, response A is better than response B," rather than on single correct answers. **RLHF** (Reinforcement Learning from Human Feedback) first trains a reward model on these preferences, then optimises the LLM against it with RL (classically PPO). **DPO** (Direct Preference Optimization) skips the reward model and RL loop and optimises the LLM directly on chosen/rejected pairs with a simple classification-style loss. It usually runs after SFT and is what makes models more helpful, better-styled and safer.

## Detailed Explanation
**Why not just SFT?** For many prompts there's no single "correct" answer, but people can easily say which of two answers is better. Preferences capture qualities like helpfulness, concision, tone and harmlessness.

**Pipeline**
```
base ─► SFT ─► sample 2+ responses per prompt ─► humans/AI judge pick "chosen" vs "rejected"
                                                          │
          RLHF: train reward model ─► PPO/GRPO on LLM ◄───┤
          DPO : directly update LLM on (chosen, rejected) ◄┘
```

| | RLHF (PPO) | DPO |
|---|---|---|
| Components | Policy, reference, reward model, value model | Policy + frozen reference |
| Stability | Finicky, many hyperparameters | Stable, like SFT |
| Compute | High | Low (works with LoRA/QLoRA) |
| Data | Preference pairs (+ online sampling) | Offline preference pairs |
| Key knob | KL penalty to reference | `beta` (how far to move from reference) |

**DPO intuition:** increase the log-probability of the chosen response relative to the rejected one, measured *relative to the reference model*, so the model doesn't drift too far. `beta` (~0.1) controls that strength.

**Related methods (2024–2026):** RLAIF / Constitutional AI (AI-generated preferences guided by principles), ORPO and SimPO (reference-free variants), KTO (thumbs-up/down data instead of pairs), and **RL with verifiable rewards** (GRPO-style training on maths and code where answers can be checked automatically). The last is central to modern "reasoning" models.

**Data format (TRL):**
```json
{"prompt":[{"role":"user","content":"Explain DNS briefly"}],
 "chosen":[{"role":"assistant","content":"DNS maps names like example.com to IP addresses..."}],
 "rejected":[{"role":"assistant","content":"DNS is a thing computers use. It is very complicated and..."}]}
```

## Example
**Analogy:** SFT is a chef copying recipes. Preference tuning is a tasting panel that keeps saying "this plate is better than that one." The chef never gets a perfect recipe but learns what diners prefer.

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

model_id = "my-org/my-sft-model"           # start from your SFT checkpoint
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
ds = load_dataset("json", data_files="prefs.jsonl", split="train")

trainer = DPOTrainer(
    model=model, processing_class=tok, train_dataset=ds,
    args=DPOConfig(output_dir="dpo-out", beta=0.1, learning_rate=5e-6,
                   num_train_epochs=1, bf16=True),
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                           lora_dropout=0.05, task_type="CAUSAL_LM"),
)
trainer.train()   # watch rewards/accuracies rise, e.g. 0.5 -> 0.8
```
With LoRA, the frozen base acts as the reference model automatically, so there's no second model copy in memory.

## Common Mistakes / Misconceptions
- Skipping SFT and running DPO on a base model. DPO works best starting from a decent SFT model.
- Low-quality or noisy pairs (chosen barely better than rejected), which teach little or teach the wrong thing.
- Too high an LR or too low a `beta`, which leads to verbose, sycophantic or degenerate outputs ("reward hacking").
- Thinking RLHF adds knowledge. Like SFT, it mainly shapes behaviour.

## Key Takeaways
- Preference tuning learns from "A is better than B" comparisons.
- RLHF = reward model + RL; DPO = direct, simpler, SFT-like training on pairs.
- It usually follows SFT and drives helpfulness, style and safety.
- DPO plus LoRA is the practical choice for most teams; verifiable-reward RL powers reasoning models.

## Related Questions
- [What is supervised fine-tuning (SFT)?](04-What-is-supervised-fine-tuning-SFT.md)
- [What are training examples and labels?](06-What-are-training-examples-and-labels.md)
- [How do you evaluate a fine-tuned model?](14-How-do-you-evaluate-a-fine-tuned-model.md)
- [What is model pretraining?](01-What-is-model-pretraining.md)
