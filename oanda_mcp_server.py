"""
OANDA Universal Backtesting MCP Server
Allows the AI to import only backtrader and math; otherwise sandboxed.
Redirects stdout during backtest to prevent protocol corruption.
Returns Sharpe ratio, total trades, total return %, win rate, avg win/loss.
"""
import asyncio, os, json, logging, sys, io
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.accounts as accounts

import backtrader as bt
import pandas as pd
import math

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
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

# ---------- Restricted import function ----------
ALLOWED_MODULES = {"backtrader", "bt", "math"}

def safe_import(name, *args, **kwargs):
    base = name.split('.')[0]
    if base in ALLOWED_MODULES:
        return __import__(name, *args, **kwargs)
    raise ImportError(f"Import of '{name}' is not allowed")

SAFE_BUILTINS = {
    "__import__": safe_import,
    "__build_class__": __build_class__,
    "__name__": "__main__",
    "__doc__": "",
    "__module__": "__main__",
    "__qualname__": "UserStrategy",
    "True": True, "False": False, "None": None,
    "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
    "bytes": bytes, "callable": callable, "chr": chr, "complex": complex,
    "divmod": divmod, "enumerate": enumerate, "filter": filter, "float": float,
    "format": format, "frozenset": frozenset, "getattr": getattr,
    "globals": lambda: {}, "hasattr": hasattr, "hash": hash, "hex": hex,
    "id": id, "int": int, "isinstance": isinstance, "issubclass": issubclass,
    "iter": iter, "len": len, "list": list, "locals": lambda: {},
    "map": map, "max": max, "min": min, "next": next, "object": object,
    "oct": oct, "ord": ord, "pow": pow, "print": print, "range": range,
    "repr": repr, "reversed": reversed, "round": round, "set": set,
    "slice": slice, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "type": type, "vars": vars, "zip": zip,
}

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

# ---------- Safe code execution with stdout redirect ----------
def run_user_strategy(df: pd.DataFrame, code: str) -> dict:
    local_namespace = {}
    try:
        exec(code, {"bt": bt, "math": math, "__builtins__": SAFE_BUILTINS}, local_namespace)
    except Exception as e:
        return {"error": f"Strategy code compilation failed: {e}"}

    UserStrategy = local_namespace.get("UserStrategy")
    if not UserStrategy or not issubclass(UserStrategy, bt.Strategy):
        return {"error": "Code must define a 'UserStrategy' subclass of bt.Strategy"}

    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.addstrategy(UserStrategy)
    start_cash = 10000.0
    cerebro.broker.setcash(start_cash)
    cerebro.broker.setcommission(commission=0.0001)

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0, annualize=True)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    old_stdout = sys.stdout
    fake_stdout = io.StringIO()
    sys.stdout = fake_stdout
    try:
        results = cerebro.run()
    except Exception as e:
        sys.stdout = old_stdout
        return {"error": f"Backtest runtime error: {e}"}
    finally:
        sys.stdout = old_stdout
        fake_stdout.getvalue()   # discard

    strat = results[0]
    end_value = cerebro.broker.getvalue()

    # Sharpe
    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    sharpe = sharpe_analysis.get('sharperatio')
    if sharpe is None:
        sharpe = 0.0

    # Total return percentage
    total_return_pct = round(((end_value - start_cash) / start_cash) * 100, 2)

    # Trade analysis
    trade_analysis = strat.analyzers.trades.get_analysis()
    total_closed = trade_analysis.get('total', {}).get('closed', 0) if isinstance(trade_analysis, dict) else 0

    won = trade_analysis.get('won', {}) if isinstance(trade_analysis, dict) else {}
    lost = trade_analysis.get('lost', {}) if isinstance(trade_analysis, dict) else {}
    won_total = won.get('total', 0) if isinstance(won, dict) else 0
    lost_total = lost.get('total', 0) if isinstance(lost, dict) else 0
    avg_win = won.get('average', 0) if isinstance(won, dict) else 0
    avg_loss = lost.get('average', 0) if isinstance(lost, dict) else 0

    # Win rate
    if total_closed > 0:
        win_rate = round((won_total / total_closed) * 100, 1)
    else:
        win_rate = 0.0

    return {
        "sharpe": round(float(sharpe), 4),
        "total_trades": total_closed,
        "total_return_pct": total_return_pct,
        "win_rate": win_rate,
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4)
    }

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
            description="Backtest a user-provided Python strategy (bt.Strategy subclass) on OANDA data and return performance metrics.",
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
