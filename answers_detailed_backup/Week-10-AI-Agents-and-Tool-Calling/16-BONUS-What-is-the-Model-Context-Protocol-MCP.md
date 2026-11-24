# What is the Model Context Protocol (MCP)?

> Week 10 – AI Agents and Tool Calling · Bonus question (added) · BONUS

## Short Answer
The Model Context Protocol (MCP) is an open standard, introduced by Anthropic in late 2024 and now widely adopted across vendors. It defines how AI applications connect to external tools and data. An **MCP server** exposes tools, resources and prompts. An **MCP client**, inside a host app such as Claude Desktop, Claude Code or an IDE, discovers and calls them over a common JSON-RPC protocol. It is often called "USB-C for AI": build an integration once and use it with any MCP-compatible agent.

## Detailed Explanation
**The problem it solves.** Before MCP, every app × every tool needed custom glue (an N×M problem). With MCP, each tool ships one server and each app implements one client (N+M).

```
 Host app (Claude Code, IDE, your agent)
   ├─ MCP client ──stdio──────▶ MCP server: filesystem
   ├─ MCP client ──HTTP───────▶ MCP server: GitHub (remote, OAuth)
   └─ MCP client ──HTTP───────▶ MCP server: your Postgres
                                   exposes: tools / resources / prompts
```

**Primitives:**

| Primitive | Controlled by | Example |
|---|---|---|
| Tools | Model (it decides to call them) | `create_issue(title, body)` |
| Resources | Application (data to read into context) | `file:///README.md`, DB schema |
| Prompts | User (reusable templates) | "/summarize-pr" |

**Transports:** `stdio` for local servers, and Streamable HTTP for remote servers, with OAuth-based authorization.

**How it relates to tool calling.** MCP does *not* replace the model's tool-calling ability. The host lists MCP tools to the model as ordinary tool definitions. When the model emits `tool_use`, the client forwards the call to the right MCP server and returns the result as a `tool_result`. MCP standardises the **plumbing**, while tool calling remains the **model behaviour**.

**Security (important):**
- Install only trusted servers, and pin their versions. Tool descriptions can carry prompt injection ("tool poisoning").
- Apply least privilege: scoped OAuth tokens and read-only modes where possible.
- Allowlist which MCP tools each agent may use, and require approval for write or destructive tools.
- Sandbox local servers that run code or access the filesystem.

## Example
**Analogy:** USB-C. Laptop makers don't build a special port for every device. Any USB-C device works with any USB-C laptop. MCP does the same for AI apps and tools.

A minimal MCP server using the official Python SDK (`pip install mcp`):

```python
# weather_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

@mcp.tool()
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return {"Paris": "18°C, light rain"}.get(city, f"No data for {city}")

if __name__ == "__main__":
    mcp.run()          # stdio transport by default
```

Register it with an MCP host, for example Claude Code (`claude mcp add weather -- python weather_server.py`) or a desktop app's config. When you ask "What's the weather in Paris?", the host offers `get_weather` to the model, the model emits `tool_use`, the host routes the call to your server, and the reply comes back:
```
# -> "It's 18°C with light rain in Paris."
```
The same server works unchanged with any other MCP-compatible client.

## Common Mistakes / Misconceptions
- **"MCP is a model feature."** It is a protocol between apps and tool servers. The model just sees tools.
- **"MCP replaces function calling."** It builds on tool calling. It doesn't replace it.
- **Installing random community servers** with broad credentials. They are code running with your permissions.
- **Exposing dozens of servers at once.** Too many tools hurts tool selection and wastes tokens.

## Key Takeaways
- MCP is an open, cross-vendor standard for connecting AI apps to tools, data and prompts.
- Client–server architecture; stdio or HTTP transports; tools, resources and prompts.
- It standardises integration plumbing. The model still uses ordinary tool calling.
- Treat MCP servers as privileged code: vet, pin, scope, allowlist and sandbox them.

## Related Questions
- [What is tool calling?](03-What-is-tool-calling.md)
- [What is function calling?](04-What-is-function-calling.md)
- [How does an LLM decide which tool to use?](05-How-does-an-LLM-decide-which-tool-to-use.md)
- [How do you prevent an agent from performing unauthorized operations?](14-How-do-you-prevent-an-agent-from-performing-unauthorized-operations.md)
