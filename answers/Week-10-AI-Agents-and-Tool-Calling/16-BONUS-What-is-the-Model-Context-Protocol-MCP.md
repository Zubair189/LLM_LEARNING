# What is the Model Context Protocol (MCP)?

> Week 10 – AI Agents and Tool Calling · Bonus question (added) · BONUS

## In One Line
MCP is an **open standard way to plug tools and data into AI apps**: build a connection once, and any MCP-compatible AI app can use it.

## Simple Explanation
Before MCP, every AI app needed its own custom code to connect to each tool, like GitHub, Slack or a database. With many apps and many tools, that's a lot of duplicate work.

MCP (Model Context Protocol), introduced by Anthropic in late 2024 and now used by many companies, fixes this with one shared set of rules. It has three parts:

1. **MCP server:** a small program that offers tools and data. For example, a "weather server" or a "GitHub server".
2. **MCP host:** the AI app the user works in, such as Claude Desktop, Claude Code or a code editor.
3. **MCP client:** the connector inside the host that talks to a server.

A server can offer three kinds of things:
- **Tools:** actions the model can call, such as `get_weather`.
- **Resources:** data to read, such as files or database records.
- **Prompts:** ready-made prompt templates.

**How it connects to tool calling:**
1. The host asks the MCP server "what tools do you have?"
2. It shows those tools to the model as normal tool definitions.
3. When the model asks to use one, the client forwards the request to the server and passes the result back.

So MCP does **not** replace tool calling. The model still uses ordinary tool calling; MCP is the standard **plumbing** behind it.

## Real-Life Example
MCP is often called "**USB-C for AI**". Before USB-C, every phone had its own charger. Now one plug fits many devices. Likewise, one MCP server works with any MCP-compatible AI app.

## Code Example
A tiny MCP server using the official Python SDK's `FastMCP` (needs `pip install mcp`, so it is not run here).

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")                 # name of this MCP server

@mcp.tool()                              # turn this function into an MCP tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""   # becomes the tool description
    return {"Paris": "18°C, light rain"}.get(city, f"No data for {city}")

if __name__ == "__main__":
    mcp.run()                            # talks to the host over stdio by default
```

**What happened:** the function's name, type hints and docstring become the tool's name, input schema and description. Any MCP host (like Claude Code) can connect to this server and let the model call `get_weather`.

## Watch Out
- Only connect MCP servers you trust. A server's tools run with real permissions, and its text could contain prompt injection.
- MCP is about **connecting** tools; it doesn't make the model smarter.

## Remember
- MCP = a standard way to connect AI apps to tools and data.
- Parts: server, host, client. Servers offer tools, resources and prompts.
- It builds on tool calling; it doesn't replace it.

## Related Questions
- [What is tool calling?](03-What-is-tool-calling.md)
- [What is function calling?](04-What-is-function-calling.md)
- [How does an LLM decide which tool to use?](05-How-does-an-LLM-decide-which-tool-to-use.md)
- [How do you prevent an agent from performing unauthorized operations?](14-How-do-you-prevent-an-agent-from-performing-unauthorized-operations.md)
