# What is supervised fine-tuning (SFT)?

> Week 09 – Fine-tuning and Model Optimization · Question 4

## Short Answer
Supervised fine-tuning (SFT) trains a pretrained model on **input → ideal output** pairs, such as a user prompt and the assistant reply you want. It still uses next-token cross-entropy loss, but usually computes it **only on the response tokens**, so the model learns to produce the desired answer rather than to predict the prompt. SFT is the step that turns a base model into an instruction-following assistant, and it is the most common kind of fine-tuning practitioners do.

## Detailed Explanation
- **"Supervised"** because each example carries a human-written (or strong-model-written) target answer.
- **Data format:** chat-format JSONL. The tokenizer's **chat template** turns the messages into the special-token format the model expects.
- **Loss masking:** prompt tokens get label `-100` (ignored by the loss), and assistant tokens keep their IDs. In TRL, a dataset in prompt/completion format is masked this way by default (`completion_only_loss`). For chat `messages` data, use `assistant_only_loss=True`, which needs a chat template that marks assistant turns. Older TRL versions used `DataCollatorForCompletionOnlyLM`.
- **Typical hyperparameters (LoRA):** lr 1e-4 to 2e-4, 1–3 epochs, effective batch 16–64, max sequence length 1–4k.
- **Where it fits:** pretraining → **SFT** → preference tuning (DPO/RLHF).

```
<|user|> Summarise: ...long text...  <|assistant|> Three bullet summary <eos>
 └──────── label = -100 (masked) ─────────┘ └──── loss computed here ─────┘
```

| Aspect | Pretraining | SFT |
|---|---|---|
| Data | Raw text | Prompt/response pairs |
| Labels | Next token of everything | Next token of the response |
| Size | Trillions of tokens | ~1k–100k examples |
| Teaches | Knowledge, language | Instruction following, format |

## Example
**Analogy:** SFT is an apprenticeship with worked examples. The master shows "here's the customer request, here's exactly how I'd reply," hundreds of times, until the apprentice replies the same way.

`train.jsonl`:
```json
{"messages":[{"role":"system","content":"You are a support bot. Reply in JSON."},{"role":"user","content":"My order 123 hasn't arrived"},{"role":"assistant","content":"{\"intent\":\"late_delivery\",\"order_id\":\"123\"}"}]}
```

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

model_id = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                             device_map="auto")
ds = load_dataset("json", data_files="train.jsonl", split="train")

trainer = SFTTrainer(
    model=model, processing_class=tok, train_dataset=ds,
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                           lora_dropout=0.05, task_type="CAUSAL_LM"),
    args=SFTConfig(output_dir="sft-out", num_train_epochs=2, learning_rate=2e-4,
                   bf16=True),  # add assistant_only_loss=True if the template supports it
)
trainer.train()   # logs training loss falling, e.g. 1.8 -> 0.6
```

## Common Mistakes / Misconceptions
- Training on prompt tokens too. This wastes capacity and can make the model echo prompts.
- Using a chat template that differs from the one used at inference, so the model's output looks "broken."
- Thinking SFT alone makes a model safe and preferred by users. Preference tuning usually follows.
- Using noisy or contradictory targets. SFT copies whatever you show it, including mistakes.

## Key Takeaways
- SFT = learning from demonstration pairs with next-token loss on the answer.
- Chat-format JSONL plus the model's chat template is the standard pipeline.
- Data quality matters more than quantity; ~1k excellent examples can beat 50k noisy ones.
- In practice SFT is usually done with LoRA/QLoRA via TRL's `SFTTrainer`.

## Related Questions
- [What are training examples and labels?](06-What-are-training-examples-and-labels.md)
- [What is a training dataset?](05-What-is-a-training-dataset.md)
- [What is preference tuning (RLHF and DPO)?](16-BONUS-What-is-preference-tuning-RLHF-and-DPO.md)
- [What is QLoRA?](08-What-is-QLoRA.md)
