"""
OANDA Backtesting MCP Server – Multi‑strategy support.
Explicitly checks for API credentials and logs helpful messages on failure.
"""
import asyncio, os, json, logging
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

# Verify credentials immediately
if not OANDA_API_KEY:
    logger.error("❌ OANDA_API_KEY is not set – candle requests will fail.")
if not OANDA_ACCOUNT_ID:
    logger.error("❌ OANDA_ACCOUNT_ID is not set – candle requests will fail.")
else:
    logger.info(f"✅ OANDA account ID: {OANDA_ACCOUNT_ID}")

oanda = API(access_token=OANDA_API_KEY, environment=OANDA_ENV)

app = Server("oanda-backtest")

# ---------- OANDA candle fetching ----------
def fetch_candles(instrument: str, granularity: str = "H1", count: int = 2000) -> pd.DataFrame:
    """Try to fetch candles; return empty DataFrame on any error."""
    if not OANDA_API_KEY:
        logger.error("No OANDA API key – cannot fetch candles.")
        return pd.DataFrame()
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

# ---------- Strategy classes (unchanged) ----------
class SMACross(bt.Strategy):
    params = dict(ma_fast=10, ma_slow=30)
    def __init__(self):
        self.fast = bt.indicators.SMA(self.data.close, period=self.p.ma_fast)
        self.slow = bt.indicators.SMA(self.data.close, period=self.p.ma_slow)
        self.cross = bt.indicators.CrossOver(self.fast, self.slow)
    def next(self):
        if self.cross > 0: self.buy()
        elif self.cross < 0: self.sell()

class RSIMeanRev(bt.Strategy):
    params = dict(rsi_period=14, oversold=30, overbought=70)
    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
    def next(self):
        if self.rsi < self.p.oversold and not self.position:
            self.buy()
        elif self.rsi > self.p.overbought and self.position:
            self.close()

class MACDCross(bt.Strategy):
    params = dict(fast=12, slow=26, signal=9)
    def __init__(self):
        self.macd = bt.indicators.MACD(self.data.close,
                                       period_me1=self.p.fast,
                                       period_me2=self.p.slow,
                                       period_signal=self.p.signal)
        self.cross = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
    def next(self):
        if self.cross > 0: self.buy()
        elif self.cross < 0: self.sell()

STRATEGIES = {
    "sma_cross": SMACross,
    "rsi_reversal": RSIMeanRev,
    "macd_cross": MACDCross,
}

def run_backtest(df: pd.DataFrame, strategy_type: str, params: dict) -> dict:
    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    strat_cls = STRATEGIES.get(strategy_type)
    if strat_cls is None:
        return {"error": f"Unknown strategy type: {strategy_type}"}
    cerebro.addstrategy(strat_cls, **params)
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0001)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0, annualize=True)
    results = cerebro.run()
    strat = results[0]
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0.0)
    if sharpe is None:
        sharpe = 0.0
    return {"sharpe": round(sharpe, 4)}

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
            name="backtest_strategy",
            description="Backtest a trading strategy on OANDA historical data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "instrument": {"type": "string"},
                    "granularity": {"type": "string", "default": "H1"},
                    "strategy_type": {"type": "string", "enum": ["sma_cross", "rsi_reversal", "macd_cross"]},
                    "params": {"type": "object"}
                },
                "required": ["instrument", "strategy_type", "params"]
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

    elif name == "backtest_strategy":
        instrument = arguments["instrument"]
        granularity = arguments.get("granularity", "H1")
        strategy_type = arguments["strategy_type"]
        params = arguments["params"]
        df = fetch_candles(instrument, granularity)
        if df.empty:
            return [TextContent(type="text", text=json.dumps({"error": "No data or authentication failed"}))]
        result = run_backtest(df, strategy_type, params)
        return [TextContent(type="text", text=json.dumps(result))]

    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
