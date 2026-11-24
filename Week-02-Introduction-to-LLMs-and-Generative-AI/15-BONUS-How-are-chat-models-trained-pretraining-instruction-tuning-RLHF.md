# How are chat models trained (pretraining, instruction tuning, RLHF)?

> Week 02 – Introduction to LLMs and Generative AI · Bonus question (added)

## Short Answer
Chat models like Claude, ChatGPT or Gemini are built in stages. **Pretraining** teaches a Transformer to predict the next token on trillions of tokens of text and code. The result is a knowledgeable *base model* that continues text but doesn't reliably follow instructions. **Instruction tuning (SFT)** then fine-tunes it on curated examples of prompts paired with ideal assistant responses, so it learns the chat format and how to be helpful. Finally, **preference optimization** (RLHF, or alternatives such as DPO and AI-feedback methods like Constitutional AI), plus **RL on verifiable tasks**, refines the model toward responses that are more helpful, honest and harmless, and improves its reasoning.

## Detailed Explanation
```
 Stage 1: PRETRAINING           Stage 2: SFT / INSTRUCTION TUNING      Stage 3: PREFERENCE & RL
 trillions of tokens            ~10k–1M+ curated (prompt, response)     human/AI rankings, reward
 next-token prediction          demonstrations in chat format          models, verifiable rewards
        │                                  │                                   │
        ▼                                  ▼                                   ▼
   BASE MODEL  ───────────────────►  SFT MODEL  ──────────────────────►  CHAT / ASSISTANT MODEL
 "knows a lot,                    "follows instructions,              "more helpful, safer, better
  just continues text"             answers in assistant style"         calibrated, reasons better"
```

**Stage 1: Pretraining (self-supervised)**
- **Data:** web pages, books, code, papers, filtered and deduplicated.
- **Objective:** minimize cross-entropy on the next token.
- **Cost:** by far the most compute (huge GPU clusters, weeks to months).
- **Result:** a base model. Asked "What is the capital of France?", it might continue with "What is the capital of Germany?", because it is completing a quiz-like document rather than answering.

**Stage 2: Supervised Fine-Tuning (SFT) / instruction tuning**
- **Data:** high-quality demonstrations, often written by experts or generated and then filtered, covering Q&A, coding, summarization, refusals, multi-turn chat and tool use.
- **Objective:** the same next-token loss, applied only to the assistant's response tokens.
- **Result:** the model learns the *format* and *behaviour* of an assistant, including the special chat template with system/user/assistant roles.

**Stage 3: Preference optimization and reinforcement learning**
- **RLHF (classic recipe, InstructGPT 2022):**
  1. Sample several responses per prompt and have humans **rank** them.
  2. Train a **reward model** to predict those preferences.
  3. Optimize the LLM with RL (e.g. PPO) to maximize reward, with a **KL penalty** that keeps it close to the SFT model, so it doesn't drift or "reward hack".
- **DPO and relatives:** optimize directly on (chosen, rejected) pairs, with no separate reward model or RL loop. They are simpler and widely used.
- **RLAIF / Constitutional AI (Anthropic):** an AI model, guided by a written set of principles, provides feedback or critiques, which reduces the need for human labels on harmlessness.
- **RL with verifiable rewards:** for math, code and other checkable tasks, the reward is whether the answer or the tests are correct. This is central to training "reasoning" models that think step by step before answering.

**Other things that happen in modern post-training:** safety training and red-teaming, tool-use and agentic training, long-context extension, and distillation into smaller models.

> **Note on overlap:** Week 9 goes deeper into each stage: [What is model pretraining?](../Week-09-Fine-tuning-and-Model-Optimization/01-What-is-model-pretraining.md) and [What is supervised fine-tuning (SFT)?](../Week-09-Fine-tuning-and-Model-Optimization/04-What-is-supervised-fine-tuning-SFT.md). Those focus on doing it *yourself* (data formats, LoRA). This bonus question is the *big-picture pipeline* of how vendors produce chat models.

## Example
**Analogy:** Training a new doctor. **Pretraining** is reading the entire medical library: huge knowledge, no bedside manner. **SFT** is residency, watching senior doctors and imitating how they talk to patients. **RLHF** is feedback from patients and supervisors ("this explanation was clearer than that one"), gradually refining judgement and communication.

**Python: what the data looks like at each stage:**
```python
pretraining_doc = "The mitochondria is the powerhouse of the cell. It produces ATP via ..."

sft_example = {
    "messages": [
        {"role": "system", "content": "You are a helpful biology tutor."},
        {"role": "user", "content": "What do mitochondria do?"},
        {"role": "assistant", "content": "Mitochondria produce most of the cell's ATP ..."},
    ]
}

preference_example = {
    "prompt": "Explain mitochondria to a 10-year-old.",
    "chosen": "They're like tiny batteries inside your cells that make energy ...",
    "rejected": "Mitochondria perform oxidative phosphorylation via the electron transport chain ...",
}

# A toy reward-model style score and the DPO-style preference check
def toy_reward(answer: str) -> float:
    simple_words = {"like", "tiny", "batteries", "energy", "cells"}
    return sum(w.strip(".,") in simple_words for w in answer.lower().split())

r_chosen = toy_reward(preference_example["chosen"])
r_rejected = toy_reward(preference_example["rejected"])
print(r_chosen, r_rejected, "chosen preferred:", r_chosen > r_rejected)
# 5 0 chosen preferred: True
```
Real reward models are neural networks trained on many thousands of human or AI comparisons, not keyword counters. The data shapes above, however, are realistic.

## Common Mistakes / Misconceptions
- **"Chat ability comes from pretraining alone."** Base models just continue text. Assistant behaviour comes from post-training.
- **"RLHF teaches the model new facts."** Knowledge mostly comes from pretraining. Post-training mainly shapes behaviour, style, safety and reasoning habits.
- **"RLHF always uses PPO and human labels."** Modern pipelines mix DPO-style methods, AI feedback, and verifiable-reward RL.
- **Ignoring side effects.** Over-optimizing a reward model can cause sycophancy, verbosity or reward hacking; KL penalties and careful evaluation guard against this.

## Key Takeaways
- Three stages: pretraining (knowledge) → SFT (instruction following) → preference/RL (quality, safety, reasoning).
- Pretraining is self-supervised and the most compute-intensive; post-training is smaller but shapes behaviour.
- RLHF = human rankings → reward model → RL with a KL leash; DPO skips the reward model.
- Verifiable-reward RL powers modern reasoning models; Constitutional AI uses principle-guided AI feedback.
- Knowledge comes mostly from pretraining; behaviour mostly from post-training.

## Related Questions
- [What is reinforcement learning?](../Week-01-Introduction-to-Artificial-Intelligence/06-What-is-reinforcement-learning.md)
- [What is model pretraining?](../Week-09-Fine-tuning-and-Model-Optimization/01-What-is-model-pretraining.md)
- [What is supervised fine-tuning (SFT)?](../Week-09-Fine-tuning-and-Model-Optimization/04-What-is-supervised-fine-tuning-SFT.md)
- [What does GPT stand for?](./04-What-does-GPT-stand-for.md)
