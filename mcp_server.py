"""
MCP server for Trader.dev backtesting/optimization.
Replace dummy logic with real Trader.dev API calls when ready.
"""
import asyncio
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

app = Server("traderdev-proxy")

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="backtest_strategy",
            description="Run a backtest on Trader.dev with a given strategy definition",
            inputSchema={
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "object",
                        "description": "Full strategy specification"
                    },
                    "instrument": {"type": "string", "default": "EUR_USD"},
                    "timeframe": {"type": "string", "default": "H1"},
                    "from_date": {"type": "string", "default": "2024-01-01"},
                    "to_date": {"type": "string", "default": "2025-01-01"}
                },
                "required": ["strategy"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "backtest_strategy":
        # Dummy backtest – replace with real Trader.dev API call
        # Returns a Sharpe ratio based on strategy complexity (just for testing)
        complexity = len(str(arguments.get("strategy", {})))
        sharpe = round(1.0 + complexity * 0.05, 3)
        return [TextContent(type="text", text=str(sharpe))]

    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
