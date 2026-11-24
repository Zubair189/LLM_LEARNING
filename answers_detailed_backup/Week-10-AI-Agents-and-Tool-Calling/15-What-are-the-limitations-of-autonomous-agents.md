# What are the limitations of autonomous agents?

> Week 10 – AI Agents and Tool Calling · Question 15

## Short Answer
Autonomous agents are still unreliable over long, multi-step tasks.
- Small per-step errors compound.
- They can hallucinate facts or tool results, loop, or wander off-task.
- They are vulnerable to prompt injection.
- They are slower and costlier than single calls, and hard to test.

That is why production agents in 2026 are usually *bounded*: scoped tools, iteration and budget caps, sandboxing, and human approval for consequential actions.

## Detailed Explanation
**1. Compounding errors.** If each step succeeds 95% of the time, a 20-step task succeeds only about 0.95²⁰ ≈ **36%** of the time.

| Per-step success | 5 steps | 10 steps | 20 steps |
|---|---|---|---|
| 99% | 95% | 90% | 82% |
| 95% | 77% | 60% | 36% |
| 90% | 59% | 35% | 12% |

**2. Reasoning and planning gaps.** Agents can get stuck in loops, give up early, declare success prematurely, or misread ambiguous goals.

**3. Hallucination.** They may invent tool arguments, API fields, or even claim a tool succeeded when it failed.

**4. Context limits.** Long tasks overflow working memory. Compaction loses detail, and agents forget earlier constraints.

**5. Security.** Prompt injection through web pages, emails, files and malicious **MCP** tools. Over-privileged credentials enlarge the blast radius.

**6. Cost and latency.** Each step is a model call that resends a growing context. Multi-agent setups multiply this.

**7. Evaluation difficulty.** Agent runs are non-deterministic, with many valid paths, so you need trajectory-level evals and good tracing.

**8. Accountability.** Someone must own the outcome when an agent acts, which matters for legal and compliance reasons.

**Mitigations:**
- Narrow the scope.
- Prefer workflows where the steps are known.
- Use strong tool design and error messages.
- Set max iterations and budgets.
- Add checkpoints and verification steps (run tests, validate schemas).
- Gate risky actions with HITL.
- Apply least privilege, allowlists and sandboxing.
- Invest in observability and evals.

## Example
**Analogy:** a self-driving car that does well on highways (narrow, well-defined) but still needs a safety driver in chaotic city centres (open-ended, adversarial). Autonomy is earned task by task.

```python
# Why verification steps matter: compounding reliability
def task_success(p_step: float, steps: int) -> float:
    return p_step ** steps

for p in (0.99, 0.95, 0.90):
    print(p, [round(task_success(p, n), 2) for n in (5, 10, 20)])
# 0.99 [0.95, 0.9, 0.82]
# 0.95 [0.77, 0.6, 0.36]
# 0.9 [0.59, 0.35, 0.12]

# Practical guardrail config for a bounded agent
AGENT_LIMITS = {
    "max_iterations": 15,
    "max_cost_usd": 2.00,
    "timeout_s": 300,
    "tools_allowlist": ["search_docs", "run_tests", "create_pr"],
    "require_approval": ["create_pr"],
    "sandbox": "ephemeral-container, no outbound network",
}
```

## Common Mistakes / Misconceptions
- **Extrapolating from demos.** A great 3-step demo doesn't mean 30-step reliability.
- **Assuming a better model alone fixes reliability.** System design (tools, checks, scope) matters just as much.
- **Giving agents broad autonomy and credentials** before evaluating them on real tasks.
- **Measuring only final answers** instead of inspecting trajectories, tool usage and cost.

## Key Takeaways
- Errors compound over steps, so long autonomous runs are fragile.
- Key risks: hallucination, loops, context limits, prompt injection, cost.
- Bound agents with scope, limits, sandboxing, verification and HITL.
- Evaluate trajectories, not just outputs, and widen autonomy gradually.

## Related Questions
- [How do you prevent an agent from performing unauthorized operations?](14-How-do-you-prevent-an-agent-from-performing-unauthorized-operations.md)
- [What are human-in-the-loop systems?](13-What-are-human-in-the-loop-systems.md)
- [What is a multi-agent system?](10-What-is-a-multi-agent-system.md)
- [How do you evaluate a fine-tuned model?](../Week-09-Fine-tuning-and-Model-Optimization/14-How-do-you-evaluate-a-fine-tuned-model.md)
