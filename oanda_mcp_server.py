"""
OANDA Universal Backtesting MCP Server
Accepts arbitrary Python strategy code (a bt.Strategy subclass) from the LLM,
backtests it on OANDA data, and returns the Sharpe ratio.
"""
import asyncio, os, json, logging, sys, io
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

# OANDA
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.accounts as accounts

# Backtrader
import backtrader as bt
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oanda-mcp")

OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ENV = os.getenv("OANDA_ENV", "practice")

if not OANDA_API_KEY or not OANDA_ACCOUNT_ID:
    logger.error("❌ OANDA credentials missing")
else:
    logger.info("✅ OANDA credentials present")

oanda = API(access_token=OANDA_API_KEY, environment=OANDA_ENV)

app = Server("oanda-backtest")

# ---------- OANDA candle fetching ----------
def fetch_candles(instrument: str, granularity: str = "H1", count: int = 2000) -> pd.DataFrame:
    params = {"granularity": granularity, "count": count, "price": "M"}
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    try:
        resp = oanda.request(r)
    except Exception as e:
        logger.error(f"Candle fetch failed: {e}")
        return pd.DataFrame()
    candles = resp.get("candles", [])
    rows = []
    for c in candles:
        if c["complete"]:
            mid = c["mid"]
            rows.append({
                "datetime": c["time"],
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
                "volume": int(c.get("volume", 0))
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
    return df

# ---------- Safe code execution ----------
def run_user_strategy(df: pd.DataFrame, code: str) -> dict:
    """
    Execute the provided Python code which must define a class named 'UserStrategy'
    that is a subclass of bt.Strategy. Backtest it and return the Sharpe ratio.
    """
    local_namespace = {}
    try:
        # Compile and exec the code in a restricted namespace
        exec(code, {"bt": bt, "__builtins__": {}}, local_namespace)
    except Exception as e:
        return {"error": f"Strategy code compilation failed: {e}"}

    UserStrategy = local_namespace.get("UserStrategy")
    if not UserStrategy or not issubclass(UserStrategy, bt.Strategy):
        return {"error": "Code must define a 'UserStrategy' subclass of bt.Strategy"}

    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.addstrategy(UserStrategy)
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0001)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0, annualize=True)

    try:
        results = cerebro.run()
    except Exception as e:
        return {"error": f"Backtest runtime error: {e}"}

    strat = results[0]
    analysis = strat.analyzers.sharpe.get_analysis()
    sharpe = analysis.get('sharperatio')
    if sharpe is None:
        sharpe = 0.0
    return {"sharpe": round(float(sharpe), 4)}

# ---------- MCP tools ----------
@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="list_instruments",
            description="List all tradeable instruments on the OANDA account",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="backtest_python_strategy",
            description="Backtest a user-provided Python strategy (bt.Strategy subclass) on OANDA data and return the Sharpe ratio.",
            inputSchema={
                "type": "object",
                "properties": {
                    "instrument": {"type": "string", "description": "e.g. EUR_USD"},
                    "granularity": {"type": "string", "default": "H1"},
                    "code": {"type": "string", "description": "Full Python code defining a 'UserStrategy' class inheriting from bt.Strategy"}
                },
                "required": ["instrument", "code"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "list_instruments":
        try:
            r = accounts.AccountInstruments(accountID=OANDA_ACCOUNT_ID)
            resp = oanda.request(r)
            names = [i["name"] for i in resp.get("instruments", [])]
            return [TextContent(type="text", text=json.dumps(names))]
        except Exception as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    elif name == "backtest_python_strategy":
        instrument = arguments["instrument"]
        granularity = arguments.get("granularity", "H1")
        code = arguments["code"]

        df = fetch_candles(instrument, granularity)
        if df.empty:
            return [TextContent(type="text", text=json.dumps({"error": "No OANDA data (check instrument name or API key)"}))]

        result = run_user_strategy(df, code)
        return [TextContent(type="text", text=json.dumps(result))]

    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
