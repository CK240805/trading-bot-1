import asyncio
import os
import json
import requests
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

app = Server("traderdev-proxy")

# Load Trader.dev configuration from environment
TRADERDEV_API_KEY = os.getenv("TRADERDEV_API_KEY")
TRADERDEV_BACKTEST_URL = os.getenv(
    "TRADERDEV_BACKTEST_URL",
    "https://api.trader.dev/v1/backtest"   # 👈 replace with the real endpoint
)

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="backtest_strategy",
            description="Run a backtest on Trader.dev and return the Sharpe ratio.",
            inputSchema={
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "object",
                        "description": "Complete strategy definition in Trader.dev's expected format"
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
        # Build the request payload – ADAPT THIS TO MATCH TRADER.DEV'S API
        payload = {
            "strategy": arguments["strategy"],
            "instrument": arguments.get("instrument", "EUR_USD"),
            "timeframe": arguments.get("timeframe", "H1"),
            "start_date": arguments.get("from_date", "2024-01-01"),
            "end_date": arguments.get("to_date", "2025-01-01"),
            # add any other required fields your API expects
        }

        headers = {
            "Authorization": f"Bearer {TRADERDEV_API_KEY}",
            "Content-Type": "application/json"
        }

        try:
            # Call Trader.dev
            response = requests.post(
                TRADERDEV_BACKTEST_URL,
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            # Extract Sharpe ratio – ADAPT THIS PATH TO THE ACTUAL RESPONSE
            # Example: data["performance"]["sharpe_ratio"]
            sharpe = data.get("performance", {}).get("sharpe_ratio", 0.0)
            return [TextContent(type="text", text=str(sharpe))]

        except Exception as e:
            # Return an error as a zero Sharpe so the loop continues
            return [TextContent(type="text", text=f"0.0 (error: {str(e)})")]

    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
