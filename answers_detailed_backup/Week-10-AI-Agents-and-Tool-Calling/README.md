# Week 10 – AI Agents and Tool Calling

## Overview
This week moves from LLMs that *answer* to agents that *act*. You will learn how tool calling (also called function calling) lets a model request structured actions, and how your code runs those actions in an execution loop. You will also see how the Model Context Protocol (MCP) standardises tool integrations. Memory is covered in both forms: short-term (the context window) and long-term (external stores). From there the week compares fixed workflows with autonomous agents and multi-agent systems. It ends with what makes agents safe and reliable in production: error handling, human-in-the-loop approval, least-privilege permissions, allowlists, sandboxing, iteration limits, and an honest look at agents' current limitations.

## Questions
1. [What is an AI agent?](01-What-is-an-AI-agent.md)
2. [What is the difference between an LLM and an AI agent?](02-What-is-the-difference-between-an-LLM-and-an-AI-agent.md)
3. [What is tool calling?](03-What-is-tool-calling.md)
4. [What is function calling?](04-What-is-function-calling.md) *(largely a synonym of Q3; see the nuance)*
5. [How does an LLM decide which tool to use?](05-How-does-an-LLM-decide-which-tool-to-use.md)
6. [What is agent memory?](06-What-is-agent-memory.md)
7. [What is short-term memory?](07-What-is-short-term-memory.md)
8. [What is long-term memory?](08-What-is-long-term-memory.md)
9. [What is an agent workflow?](09-What-is-an-agent-workflow.md)
10. [What is a multi-agent system?](10-What-is-a-multi-agent-system.md)
11. [What is an agent execution loop?](11-What-is-an-agent-execution-loop.md)
12. [What happens when a tool call fails?](12-What-happens-when-a-tool-call-fails.md)
13. [What are human-in-the-loop systems?](13-What-are-human-in-the-loop-systems.md)
14. [How do you prevent an agent from performing unauthorized operations?](14-How-do-you-prevent-an-agent-from-performing-unauthorized-operations.md)
15. [What are the limitations of autonomous agents?](15-What-are-the-limitations-of-autonomous-agents.md)
16. **BONUS:** [What is the Model Context Protocol (MCP)?](16-BONUS-What-is-the-Model-Context-Protocol-MCP.md)

## Practical Exercise (1–2 hours)
**Build a safe "trip helper" agent with the Anthropic Python SDK.**
1. Define two tools with clear descriptions and JSON schemas:
   - `get_weather(city)`, a stub dict is fine.
   - `book_hotel(city, nights)`, a stub that just returns a confirmation ID.
2. Write the manual execution loop. Call `client.messages.create(model="claude-sonnet-5-5", ...)` and loop while `resp.stop_reason == "tool_use"`. Append `resp.content` as the assistant turn, then append the `tool_result` blocks as the user turn.
3. Add guardrails:
   - A `REGISTRY` allowlist.
   - `MAX_ITERS = 8`.
   - Argument validation, so unknown cities return `is_error: True`.
   - A human approval prompt (`input()`) before `book_hotel`.
4. Test these prompts:
   - "Weather in Paris and Rome?" (expect parallel tool calls).
   - "Weather in Atlantis?" (expect error recovery).
   - "Book 2 nights in the warmer city" (expect a multi-step run plus approval). Reject the approval and watch how the model responds.
5. Stretch goal: move `get_weather` into a FastMCP server (see the bonus question) and connect it to an MCP host such as Claude Code.
6. Log each step (tool, inputs, result, tokens) and write three sentences on what failed and why.

## Explain Aloud
Practise explaining each of these out loud in about 60 seconds, without notes:
1. **The agent execution loop.** Who decides, who executes, how results flow back via `tool_use_id`, and why you need a max-iteration cap.
2. **Tool calling vs. function calling vs. MCP.** Same model capability under two names, and MCP as the standard plumbing for exposing tools to any app.
3. **Defence in depth for agents.** Why prompts aren't security, and how allowlists, least privilege, sandboxing and human approval limit the blast radius of prompt injection.
