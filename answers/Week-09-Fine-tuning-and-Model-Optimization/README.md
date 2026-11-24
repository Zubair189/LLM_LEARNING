# Week 09 – Fine-tuning and Model Optimization

This week is about taking a model that is already trained and adapting it to your own needs, without spending a fortune. First you'll see the difference between pretraining (the huge, expensive first stage) and fine-tuning (a short extra training on your own examples), and when fine-tuning is worth it compared with simply writing a better prompt or using RAG. Then you'll learn how fine-tuning datasets and labels are built.

The core of the week is memory. A 7B model needs about 14 GB in normal 16-bit form and about 3.5 GB in 4-bit form. Changing every weight (full fine-tuning) needs about 112 GB, while QLoRA fits in under 12 GB. The week ends with how to test a fine-tuned model, how fine-tuning can make a model forget old skills, and preference tuning (RLHF and DPO). By the end you should be able to fine-tune a 7B model on a single GPU and show that it really got better.

## Questions
1. [What is model pretraining?](01-What-is-model-pretraining.md)
2. [What is fine-tuning?](02-What-is-fine-tuning.md)
3. [When should you fine-tune a model?](03-When-should-you-fine-tune-a-model.md)
4. [What is supervised fine-tuning (SFT)?](04-What-is-supervised-fine-tuning-SFT.md)
5. [What is a training dataset?](05-What-is-a-training-dataset.md) (fine-tuning focus; general definition in [Week 01](../Week-01-Introduction-to-Artificial-Intelligence/10-What-is-a-training-dataset.md))
6. [What are training examples and labels?](06-What-are-training-examples-and-labels.md)
7. [What is LoRA?](07-What-is-LoRA.md)
8. [What is QLoRA?](08-What-is-QLoRA.md)
9. [What is quantization?](09-What-is-quantization.md) (general concept)
10. [What is 4-bit quantization?](10-What-is-4-bit-quantization.md) (4-bit formats: NF4, GPTQ, AWQ, GGUF)
11. [What is full fine-tuning?](11-What-is-full-fine-tuning.md)
12. [How does LoRA reduce memory requirements?](12-How-does-LoRA-reduce-memory-requirements.md)
13. [What is catastrophic forgetting?](13-What-is-catastrophic-forgetting.md)
14. [How do you evaluate a fine-tuned model?](14-How-do-you-evaluate-a-fine-tuned-model.md)
15. [Can RAG replace fine-tuning?](15-Can-RAG-replace-fine-tuning.md)
16. **BONUS:** [What is preference tuning (RLHF and DPO)?](16-BONUS-What-is-preference-tuning-RLHF-and-DPO.md)

## Practical Exercise (1–2 hours)
**QLoRA fine-tune a small model to output structured JSON, and prove that it helped.**
1. **Data (20 min):** write or generate 300 chat-format JSONL examples. Each user message is a short customer email, and each assistant reply is JSON `{"intent": ..., "order_id": ..., "urgency": ...}`. Dedupe, then split 90/5/5 into train/val/test.
2. **Baseline (10 min):** run `Qwen/Qwen2.5-1.5B-Instruct` (or any ≤3B instruct model) with a good prompt on the test set. Record JSON-valid % and exact-match % per field.
3. **Train (30–40 min, free Colab T4 or any 16 GB GPU):** load in 4-bit with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)` (use float16 on a T4). Add `LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"], lora_dropout=0.05, task_type="CAUSAL_LM")`, train for 2 epochs with TRL `SFTTrainer`, and log `print_trainable_parameters()` and peak GPU memory.
4. **Evaluate (15 min):** re-run the same test set and compare with the baseline. Also ask 5 unrelated general questions before and after, to look for catastrophic forgetting.
5. **Write-up (10 min):** a table covering baseline vs fine-tuned metrics, trainable params, adapter size on disk and peak memory, plus one paragraph on whether RAG or structured outputs could have solved this instead.

## Explain Aloud
Practise explaining each of these in under 2 minutes, without notes:
1. **LoRA's memory savings.** Why freezing the model and training only a small add-on (r=16: about 131 thousand numbers instead of 16.8 million per 4096 × 4096 table) cuts training memory for a 7B model from about 112 GB to about 16–20 GB, and what it does *not* save.
2. **QLoRA and 4-bit quantization.** How 4-bit (NF4) stores each weight in about half a byte (7B ≈ 3.5 GB), why the small adapters stay in normal 16-bit, and what you give up in speed and quality.
3. **Fine-tuning vs RAG.** Fine-tuning changes behaviour, RAG adds knowledge; when each one wins, and why real systems often use both.
