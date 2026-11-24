# Why do LLMs sometimes hallucinate?

> Week 02 – Introduction to LLMs and Generative AI · Question 12

## Short Answer
LLMs hallucinate, meaning they produce fluent but false or made-up content, because they are trained to generate **plausible next tokens**, not to check facts against a source of truth. When the model lacks the knowledge (a rare fact, something after its training cutoff, your private data), it still produces the statistically likely-sounding continuation instead of saying "I don't know". Other causes include gaps and errors in the training data, training rewards that favour confident answers over admitting uncertainty, sampling randomness, and ambiguous prompts. Hallucination can be greatly reduced through grounding (RAG, tools), good prompting and verification, but not eliminated.

## Detailed Explanation
**Root causes:**
| Cause | Explanation | Example |
|---|---|---|
| Next-token objective | Optimized for *likely* text, not *true* text | Invents a plausible-looking citation with real-sounding authors |
| Knowledge gaps / cutoff | No information about recent or niche facts | Describes "the latest version" of a library incorrectly |
| Compressed, lossy memory | Billions of facts squeezed into weights; rare facts get blurred | Wrong birth year for a minor historical figure |
| Training incentives | Benchmarks and graders historically rewarded guessing over abstaining | Answers confidently instead of "I'm not sure" |
| Sampling | High temperature picks lower-probability (possibly wrong) tokens | Different, inconsistent answers across runs |
| Prompt issues | False premises, ambiguity, pressure to answer | "Why did Einstein win the 1950 Nobel Prize?" (he won the 1921 prize) |
| Long or noisy context | Relevant info is buried or contradicted | Mixes up two similar documents in a RAG prompt |

**Types of hallucination:**
- **Factual:** wrong facts about the world.
- **Faithfulness (intrinsic):** contradicts the source document you provided (a summary that adds claims).
- **Fabricated references:** fake URLs, papers, API functions or package names. The last of these is a security risk ("slopsquatting").

**Mitigations (engineering is the answer):**
1. **Ground it:** supply sources via RAG or tool calls (search, databases) and instruct the model to answer *only* from them (Week 8).
2. **Allow abstention:** tell it explicitly that "I don't know" is acceptable.
3. **Ask for citations or quotes** from the provided context, then verify them programmatically.
4. **Lower temperature** for factual tasks; use structured outputs.
5. **Verify:** a second LLM pass or rule checks, a self-consistency vote, or human review for high-stakes output (Week 11).
6. **Use stronger, newer models.** Hallucination rates have dropped noticeably, but they are not zero.

> **Note on overlap:** this question covers *why* hallucination happens in general. Week 8's [What is hallucination in RAG?](../Week-08-Retrieval-Augmented-Generation-RAG/12-What-is-hallucination-in-RAG.md) covers hallucinations *despite* retrieved context (faithfulness failures). Week 11's [What is hallucination detection?](../Week-11-Production-LLM-Applications/14-What-is-hallucination-detection.md) covers *detecting* them in production.

## Example
**Analogy:** A confident student in an oral exam who doesn't know the answer but keeps talking in the right style and vocabulary. It sounds convincing, and the details are invented.

**Python: grounding plus permission to abstain (Anthropic SDK):**
```python
import anthropic
client = anthropic.Anthropic()

context = "Acme's refund window is 30 days. Refunds go to the original payment method."
resp = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=200,
    system=("Answer ONLY using the provided context. If the answer is not in the "
            "context, reply exactly: I don't know based on the provided information."),
    messages=[{"role": "user",
               "content": f"<context>{context}</context>\n\nDoes Acme offer store credit?"}],
)
print(resp.content[0].text)
# I don't know based on the provided information.
```

**Python: a cheap programmatic check that quoted evidence really exists in the source:**
```python
def quotes_supported(answer_quotes, source):
    return {q: (q in source) for q in answer_quotes}

source = "Acme's refund window is 30 days. Refunds go to the original payment method."
print(quotes_supported(["refund window is 30 days", "refund window is 60 days"], source))
# {'refund window is 30 days': True, 'refund window is 60 days': False}
```

## Common Mistakes / Misconceptions
- **"Hallucinations are random bugs that will be patched."** They follow from how LLMs are trained. They can be reduced but not removed.
- **"RAG eliminates hallucinations."** Models can still misread or ignore retrieved context.
- **"Temperature 0 means no hallucinations."** Greedy decoding can still confidently produce wrong facts.
- **"The model knows when it's lying."** It isn't lying. It has no reliable built-in truth check, though newer models are better calibrated about uncertainty.

## Key Takeaways
- LLMs optimize for plausible text, not verified truth, so gaps get filled with fluent guesses.
- Causes include knowledge gaps and cutoff, lossy memorization, training incentives, sampling and prompts.
- Reduce hallucinations with grounding (RAG/tools), permission to abstain, citations, verification and lower temperature.
- Always validate high-stakes outputs.

## Related Questions
- [What is hallucination in RAG?](../Week-08-Retrieval-Augmented-Generation-RAG/12-What-is-hallucination-in-RAG.md)
- [What is hallucination detection?](../Week-11-Production-LLM-Applications/14-What-is-hallucination-detection.md)
- [Why do LLMs need external knowledge?](../Week-08-Retrieval-Augmented-Generation-RAG/02-Why-do-LLMs-need-external-knowledge.md)
- [What is a language model?](./05-What-is-a-language-model.md)
