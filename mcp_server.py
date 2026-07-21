import asyncio
import os
import requests
from mcp.server import Server
from mcp.server.stdio import stdio_server

app = Server("traderdev-proxy")

# Trader.dev API configuration – set these as environment variables
TRADERDEV_API_KEY = os.getenv("TRADERDEV_API_KEY")
TRADERDEV_BACKTEST_URL = os.getenv("TRADERDEV_BACKTEST_URL", "https://api.trader.dev/v1/backtest")

@app.list_tools()
async def list_tools():
    return [
        {
            "name": "backtest_strategy",
            "description": "Run a backtest on Trader.dev with a given strategy definition",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "object",
                        "description": "Full strategy specification in Trader.dev's expected format"
                    },
                    "instrument": {"type": "string", "default": "EUR_USD"},
                    "timeframe": {"type": "string", "default": "H1"},
                    "from_date": {"type": "string", "default": "2024-01-01"},
                    "to_date": {"type": "string", "default": "2025-01-01"}
                },
                "required": ["strategy"]
            }
        }
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name != "backtest_strategy":
        raise ValueError(f"Unknown tool: {name}")

    # Build the request to Trader.dev
    payload = {
        "strategy": arguments["strategy"],
        "instrument": arguments.get("instrument", "EUR_USD"),
        "timeframe": arguments.get("timeframe", "H1"),
        "date_range": {
            "from": arguments.get("from_date", "2024-01-01"),
            "to": arguments.get("to_date", "2025-01-01")
        }
    }
    headers = {"Authorization": f"Bearer {TRADERDEV_API_KEY}"}
    try:
        resp = requests.post(TRADERDEV_BACKTEST_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Assuming Trader.dev returns something like {"performance": {"sharpe_ratio": 1.2}}
        sharpe = data.get("performance", {}).get("sharpe_ratio", 0.0)
        return {"sharpe": sharpe}
    except Exception as e:
        return {"sharpe": 0.0, "error": str(e)}

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
