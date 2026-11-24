# How are chat models trained (pretraining, instruction tuning, RLHF)?

> Week 02 – Introduction to LLMs and Generative AI · Bonus question (added)

## In One Line
Chat models are trained in **three stages**: read lots of text (pretraining), learn to follow instructions from examples (instruction tuning), then improve from feedback on their answers (RLHF and similar methods).

## Simple Explanation
**Stage 1: Pretraining (gain knowledge)**
- The model reads a huge amount of text and code and practises guessing the next token.
- This is by far the most expensive stage.
- The result is a **base model**. It knows a lot but only **continues text**. Ask it "What is the capital of France?" and it might reply "What is the capital of Germany?", as if continuing a quiz.

**Stage 2: Instruction tuning, also called SFT (learn to be an assistant)**
- SFT means **Supervised Fine-Tuning**: more training on carefully written examples of a question and an ideal answer.
- The model learns the **chat format** and how a helpful assistant behaves.

**Stage 3: Learning from feedback (get better and safer)**
- **RLHF** (Reinforcement Learning from Human Feedback):
  1. The model writes several answers; people **rank** them.
  2. A separate **reward model** learns to predict which answers people prefer.
  3. The chat model is trained to produce answers that score highly, while staying close to how it behaved before, so it doesn't learn silly tricks.
- **DPO** is a simpler alternative that learns directly from "better vs. worse" answer pairs.
- **AI feedback** (like Anthropic's Constitutional AI) uses an AI guided by written principles to give feedback, in place of some human labels.
- **Checkable rewards:** for maths or code, the reward is "is the answer right / do the tests pass?". This helps train step-by-step reasoning.

Week 9 goes deeper into pretraining and SFT, and how to do them yourself. This question is the big picture.

## Real-Life Example
Training a new doctor. **Pretraining** is reading the whole medical library: lots of knowledge, no bedside manner. **SFT** is copying how senior doctors talk to patients. **RLHF** is feedback like "this explanation was clearer than that one", which slowly improves their judgement.

## Code Example
What the training data looks like at each stage.

```python
# Stage 1: plain text (the model learns to guess the next token)
pretraining_text = "The mitochondria is the powerhouse of the cell. It produces energy ..."

# Stage 2: an example chat with an ideal answer
sft_example = [
    {"role": "user", "content": "What do mitochondria do?"},
    {"role": "assistant", "content": "They make most of the energy your cells use."},
]

# Stage 3: one prompt, a better answer and a worse answer
preference_example = {
    "prompt": "Explain mitochondria to a 10-year-old.",
    "chosen": "They're like tiny batteries inside your cells.",      # people preferred this
    "rejected": "They perform oxidative phosphorylation.",           # too technical for a child
}

print(len(sft_example), "chat turns")                    # 2 chat turns
print("better answer:", preference_example["chosen"])    # better answer: They're like tiny batteries inside your cells.
```

**What happened:** each stage uses a different kind of data: raw text, then example conversations, then "better vs. worse" pairs.

## Watch Out
- **Chat skills don't come from pretraining alone.** A base model just continues text.
- **RLHF mostly shapes behaviour, not knowledge.** Facts mostly come from pretraining.

## Remember
- Pretraining → knowledge.
- Instruction tuning (SFT) → follows instructions.
- RLHF and similar methods → more helpful, safer, better reasoning.

## Related Questions
- [What is reinforcement learning?](../Week-01-Introduction-to-Artificial-Intelligence/06-What-is-reinforcement-learning.md)
- [What is model pretraining?](../Week-09-Fine-tuning-and-Model-Optimization/01-What-is-model-pretraining.md)
- [What is supervised fine-tuning (SFT)?](../Week-09-Fine-tuning-and-Model-Optimization/04-What-is-supervised-fine-tuning-SFT.md)
- [What does GPT stand for?](./04-What-does-GPT-stand-for.md)
