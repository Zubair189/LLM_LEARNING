# What are training examples and labels?

> Week 09 – Fine-tuning and Model Optimization · Question 6

## Short Answer
A **training example** is one input the model sees during training, and its **label** is the correct output the model should learn to produce for that input. In LLM fine-tuning, an example is usually a whole conversation, and the labels are the token IDs of the assistant's reply. Prompt tokens are masked out (label `-100`) so that the loss only rewards generating the right answer.

## Detailed Explanation
**Classic ML vs LLMs**

| Setting | Example (input) | Label (target) |
|---|---|---|
| Spam classifier | Email text | `spam` / `ham` |
| Image classifier | Pixels | `cat` |
| LLM pretraining | Tokens `t1..tn` | Same sequence shifted: `t2..t(n+1)` |
| LLM SFT | Chat prompt + reply | Reply tokens only (prompt = -100) |
| DPO | Prompt | A *preferred* and a *rejected* reply |

**How labels work inside a causal LM**
- `input_ids` = the tokenized full conversation.
- `labels` = a copy of `input_ids`, with prompt positions set to `-100`.
- The model shifts labels by one internally, so position *i* predicts token *i+1*.
- PyTorch's cross-entropy ignores `-100`, so only reply tokens contribute to the loss.

```
tokens : [<user>] [Hi] [<asst>] [Hello] [!] [<eos>]
labels : [ -100 ] [-100] [-100] [Hello] [!] [<eos>]
```

**Good labels are**
- correct, and formatted exactly as you'd want at inference;
- consistent across examples (same style for the same kind of request);
- complete, including the end-of-sequence token so the model learns when to stop.

## Example
**Analogy:** flashcards. The front of the card (the example) is the question, the back (the label) is the answer. You only get marked on what you write for the back, not for reading the front aloud.

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
prompt = [{"role": "user", "content": "Capital of France?"}]
full = prompt + [{"role": "assistant", "content": "Paris."}]

def ids(msgs, gen=False):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)
    return tok(text, add_special_tokens=False).input_ids

prompt_ids = ids(prompt, gen=True)   # ends with the "<assistant>" header
full_ids = ids(full)

labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
print(len(full_ids), sum(l != -100 for l in labels))
# e.g. 28 5  -> only ~5 reply tokens ("Paris." + end tokens) are trained on
```

## Common Mistakes / Misconceptions
- Forgetting the EOS token in labels, so the model rambles and never stops.
- Leaving prompt tokens unmasked on long-prompt tasks. The loss then gets dominated by copying the prompt.
- Inconsistent labels (two different correct formats for the same input), which confuses the model.
- Thinking LLM pretraining has no labels. It does; they are simply the next tokens.

## Key Takeaways
- Example = input; label = desired output. In LLMs both are token sequences.
- In SFT, labels are the assistant tokens; the prompt is masked with `-100`.
- Label quality and consistency directly determine what the model learns.
- DPO-style data replaces a single label with a chosen/rejected pair.

## Related Questions
- [What is a training dataset?](05-What-is-a-training-dataset.md)
- [What is supervised fine-tuning (SFT)?](04-What-is-supervised-fine-tuning-SFT.md)
- [What is preference tuning (RLHF and DPO)?](16-BONUS-What-is-preference-tuning-RLHF-and-DPO.md)
- [What is model pretraining?](01-What-is-model-pretraining.md)
