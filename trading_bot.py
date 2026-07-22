"""
Autonomous Trading Bot – NVIDIA API (DeepSeek) + OANDA + Telegram + MCP (Trader.dev)
Dynamic watch list validated against OANDA instruments.
Trade size automatically adjusted to OANDA's minimum/maximum constraints.
Set all required environment variables before running.
"""
import os, json, time, logging, threading, asyncio
from datetime import datetime
from typing import Optional, List, Dict, Set
import requests
import schedule
from fastapi import FastAPI
import uvicorn
from telegram import Bot
from telegram.error import TelegramError

# OANDA
from oandapyV20 import API
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.accounts as accounts

# MCP client
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# NVIDIA LLM client
from openai import OpenAI

# Load .env file locally (ignored on Render)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TradingBot")

# ---------- Config from environment ----------
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ENV = os.getenv("OANDA_ENV", "practice")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
MCP_SERVER_COMMAND = os.getenv("MCP_SERVER_COMMAND", "python")
MCP_SERVER_ARGS = os.getenv("MCP_SERVER_ARGS", "mcp_server.py").split()
ALLOCATED_CAPITAL = 100.0  # hard cap in USD

# Trader.dev API (used by MCP proxy)
TRADERDEV_API_KEY = os.getenv("TRADERDEV_API_KEY")

# NVIDIA client
llm_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

OANDA_URL = "https://api-fxpractice.oanda.com" if OANDA_ENV == "practice" else "https://api-fxtrade.oanda.com"
oanda_api = API(access_token=OANDA_API_KEY, environment=OANDA_ENV)

DEFAULT_TIMEFRAME = "M5"
CURRENT_WATCHLIST: List[str] = []
VALID_INSTRUMENTS: Dict[str, dict] = {}  # name -> {minTradeSize, maxOrderUnits, pipLocation, ...}

# Duplicate Telegram startup message guard
_STARTUP_MESSAGE_SENT = False

# Default safe instruments (guaranteed to exist on most OANDA demo accounts)
DEFAULT_INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD", "US30_USD"]

# ---------- NVIDIA LLM Chat ----------
def deepseek_chat(prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        completion = llm_client.chat.completions.create(
            model="deepseek-ai/deepseek-v4-pro",
            messages=messages,
            temperature=1,
            top_p=0.95,
            max_tokens=16384,
            extra_body={"chat_template_kwargs": {"thinking": False}},
            stream=False
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM API error: {e}")
        return ""

# ---------- News Fetcher ----------
def fetch_news() -> str:
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return "No news API key set."
    url = f"https://newsapi.org/v2/top-headlines?category=business&language=en&apiKey={api_key}"
    try:
        resp = requests.get(url, timeout=10)
        articles = resp.json().get("articles", [])[:5]
        return "\n".join([f"- {a['title']}" for a in articles])
    except Exception as e:
        return f"News fetch error: {e}"

# ---------- OANDA Instrument Validation ----------
def get_valid_instruments() -> Dict[str, dict]:
    """
    Fetch the list of tradeable instruments from OANDA.
    Returns a dict mapping instrument name -> details (minTradeSize, maxOrderUnits, etc.)
    """
    try:
        r = accounts.AccountInstruments(accountID=OANDA_ACCOUNT_ID)
        resp = oanda_api.request(r)
        instruments = {}
        for instr in resp.get("instruments", []):
            instruments[instr["name"]] = {
                "minTradeSize": float(instr.get("minimumTradeSize", 1)),
                "maxOrderUnits": float(instr.get("maximumOrderUnits", 1000000)),
                "pipLocation": instr.get("pipLocation", -4),
            }
        logger.info(f"Fetched {len(instruments)} valid instruments from OANDA.")
        return instruments
    except Exception as e:
        logger.error(f"Failed to fetch OANDA instruments: {e}")
        # Fallback: provide minimal defaults for our safe list
        defaults = {}
        for name in DEFAULT_INSTRUMENTS:
            defaults[name] = {"minTradeSize": 1, "maxOrderUnits": 1000000, "pipLocation": -4}
        return defaults

def update_valid_instruments():
    global VALID_INSTRUMENTS
    VALID_INSTRUMENTS = get_valid_instruments()
    if not VALID_INSTRUMENTS:
        # Restore safe defaults if the API call fails completely
        VALID_INSTRUMENTS = {
            instr: {"minTradeSize": 1, "maxOrderUnits": 1000000, "pipLocation": -4}
            for instr in DEFAULT_INSTRUMENTS
        }

# ---------- Dynamic Watch List ----------
def update_watch_list() -> List[str]:
    global VALID_INSTRUMENTS
    if not VALID_INSTRUMENTS:
        update_valid_instruments()

    news = fetch_news()
    system = (
        "You are a professional financial market analyst. "
        "Select the best 5 instruments to trade on OANDA for the upcoming session. "
        "Return ONLY a valid JSON array of strings, e.g., ['EUR_USD','XAU_USD']. "
        "Use exact OANDA instrument names."
    )
    prompt = f"Recent news headlines:\n{news}\n\nCurrent datetime: {datetime.now().isoformat()}\n\nProvide the JSON array."
    response = deepseek_chat(prompt, system)
    if not response:
        logger.warning("LLM call failed, using default watch list.")
        return _filtered_default_watchlist()

    try:
        if "```" in response:
            snippet = response.split("```")[1]
            if snippet.startswith("json"):
                snippet = snippet[4:]
            watchlist = json.loads(snippet.strip())
        else:
            watchlist = json.loads(response.strip())

        if isinstance(watchlist, list) and all(isinstance(i, str) for i in watchlist):
            valid_watchlist = [i for i in watchlist if i in VALID_INSTRUMENTS]
            if len(valid_watchlist) < 3:
                logger.warning("Too many invalid instruments, using default.")
                return _filtered_default_watchlist()
            # Top up to 5
            while len(valid_watchlist) < 5:
                for default in DEFAULT_INSTRUMENTS:
                    if default not in valid_watchlist and default in VALID_INSTRUMENTS:
                        valid_watchlist.append(default)
                        if len(valid_watchlist) >= 5:
                            break
            logger.info(f"Updated watch list: {valid_watchlist}")
            return valid_watchlist[:5]
    except Exception as e:
        logger.error(f"Failed to parse watch list: {e}")

    logger.warning("Using default watch list.")
    return _filtered_default_watchlist()

def _filtered_default_watchlist() -> List[str]:
    if not VALID_INSTRUMENTS:
        return DEFAULT_INSTRUMENTS.copy()
    return [i for i in DEFAULT_INSTRUMENTS if i in VALID_INSTRUMENTS][:5]

# ---------- OANDA Helpers ----------
def oanda_get_prices(instruments: list) -> dict:
    params = {"instruments": ",".join(instruments)}
    r = pricing.PricingInfo(accountID=OANDA_ACCOUNT_ID, params=params)
    resp = oanda_api.request(r)
    prices = {}
    for p in resp.get("prices", []):
        if p["type"] == "PRICE":
            prices[p["instrument"]] = {
                "bid": float(p["bids"][0]["price"]),
                "ask": float(p["asks"][0]["price"])
            }
    return prices

def adjust_units_to_instrument(instrument: str, desired_units: int, price: float) -> int:
    """
    Adjust units to meet OANDA's minimum/maximum trade size and
    still keep total exposure ≤ $100.
    Returns 0 if no valid size fits the $100 cap.
    """
    info = VALID_INSTRUMENTS.get(instrument)
    if not info:
        logger.warning(f"No instrument info for {instrument}, rejecting trade.")
        return 0

    min_units = int(info["minTradeSize"])
    max_units = int(info["maxOrderUnits"])

    # Ensure absolute units are at least the minimum
    abs_units = abs(desired_units)
    if abs_units < min_units:
        abs_units = min_units
    # Cap at maximum
    if abs_units > max_units:
        abs_units = max_units

    # Calculate exposure
    exposure = abs_units * price
    # If exposure exceeds $100, scale down to the maximum allowed
    if exposure > ALLOCATED_CAPITAL:
        max_allowed_units = int(ALLOCATED_CAPITAL / price)
        if max_allowed_units < min_units:
            logger.info(f"Cannot trade {instrument}: min {min_units} units costs ${min_units*price:.2f} > $100.")
            return 0
        abs_units = min(max_allowed_units, max_units)
        # If after capping we're still below minimum, reject
        if abs_units < min_units:
            abs_units = min_units
            # Re-check exposure after setting to minimum
            if abs_units * price > ALLOCATED_CAPITAL:
                logger.info(f"Cannot trade {instrument}: min units exceeds $100.")
                return 0

    # Preserve direction
    return abs_units if desired_units >= 0 else -abs_units

def oanda_place_order(instrument: str, units: int, stop_loss: Optional[float] = None, take_profit: Optional[float] = None):
    data = {
        "order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(units),
            "timeInForce": "FOK"
        }
    }
    if stop_loss:
        data["order"]["stopLossOnFill"] = {"price": str(round(stop_loss, 5))}
    if take_profit:
        data["order"]["takeProfitOnFill"] = {"price": str(round(take_profit, 5))}
    r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=data)
    try:
        resp = oanda_api.request(r)
        logger.info(f"Order placed: {resp}")
        return resp
    except Exception as e:
        logger.error(f"Order error: {e}")
        return None

def oanda_get_open_trades():
    r = trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)
    resp = oanda_api.request(r)
    return resp.get("trades", [])

def oanda_get_account_summary():
    r = accounts.AccountSummary(accountID=OANDA_ACCOUNT_ID)
    resp = oanda_api.request(r)
    return resp["account"]

# ---------- Telegram ----------
async def send_telegram_message(text: str):
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
    except TelegramError as e:
        logger.error(f"Telegram error: {e}")

def tg_send_sync(text: str):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is None:
        asyncio.run(send_telegram_message(text))
    else:
        asyncio.ensure_future(send_telegram_message(text))

# ---------- LLM-Driven Strategy Optimizer ----------
class LLMStrategyOptimizer:
    def __init__(self):
        self.best_strategy = None
        self.best_sharpe = -9999

    async def _call_mcp_backtest(self, strategy: dict, instrument="EUR_USD", timeframe="H1") -> float:
        server_params = StdioServerParameters(command=MCP_SERVER_COMMAND, args=MCP_SERVER_ARGS)
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                if not any(t.name == "backtest_strategy" for t in tools):
                    logger.error("MCP server missing 'backtest_strategy' tool.")
                    return 0.0
                result = await session.call_tool(
                    "backtest_strategy",
                    arguments={
                        "strategy": strategy,
                        "instrument": instrument,
                        "timeframe": timeframe
                    }
                )
                if result.content and len(result.content) > 0:
                    sharpe_str = result.content[0].text
                    return float(sharpe_str)
                return 0.0

    async def run_optimization_loop(self):
        system = (
            "You are an expert quantitative trading strategist. "
            "Output a valid JSON strategy for EUR/USD H1. "
            "Use only JSON, no other text."
        )
        user = "Propose an initial trading strategy."
        response = deepseek_chat(user, system)
        if not response:
            return
        try:
            current_strategy = json.loads(response)
        except:
            return

        for i in range(10):
            sharpe = await self._call_mcp_backtest(current_strategy)
            logger.info(f"Iteration {i+1} Sharpe = {sharpe:.3f}")
            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                self.best_strategy = current_strategy
                if sharpe >= 2.0:
                    logger.info("Amazing strategy found!")
                    break
            improvement_prompt = f"Improve this strategy (Sharpe {sharpe:.3f}): {json.dumps(current_strategy)}"
            response = deepseek_chat(improvement_prompt, system)
            if not response:
                break
            try:
                current_strategy = json.loads(response)
            except:
                break

llm_strategy_optimizer = LLMStrategyOptimizer()

def mcp_optimization_runner():
    try:
        asyncio.run(llm_strategy_optimizer.run_optimization_loop())
    except Exception as e:
        logger.error(f"Optimization loop failed: {e}")

# ---------- Morning Analysis Job ----------
def morning_analysis():
    global CURRENT_WATCHLIST
    CURRENT_WATCHLIST = update_watch_list()
    if not CURRENT_WATCHLIST:
        tg_send_sync("🌅 Morning analysis skipped – watch list is empty.")
        return
    prices = oanda_get_prices(CURRENT_WATCHLIST)
    if not prices:
        tg_send_sync("🌅 OANDA data unavailable.")
        return
    news = fetch_news()
    prompt = f"Market snapshot (bid/ask):\n{json.dumps(prices, indent=2)}\nNews:\n{news}\nProvide a concise morning outlook."
    analysis = deepseek_chat(prompt, "You are a professional financial market analyst.")
    if not analysis:
        tg_send_sync("🌅 Morning analysis failed – LLM unavailable.")
        return
    message = f"🌅 *Morning Market Analysis* ({datetime.now().strftime('%Y-%m-%d')})\n\n🎯 Watch list: {', '.join(CURRENT_WATCHLIST)}\n\n{analysis}"
    tg_send_sync(message)

# ---------- Night Performance Report ----------
def night_performance():
    try:
        summary = oanda_get_account_summary()
    except Exception as e:
        tg_send_sync(f"🌙 Night report failed: {e}")
        return
    trades_list = oanda_get_open_trades()
    pnl = float(summary.get("unrealizedPL", 0))
    balance = float(summary.get("balance", 0))
    nav = float(summary.get("NAV", 0))
    open_trades_text = "\n".join([f"{t['instrument']} {t['currentUnits']} @ {t['price']}" for t in trades_list])
    message = f"""🌙 *Night Performance Report* ({datetime.now().strftime('%Y-%m-%d')})

- Balance: ${balance:.2f}
- Unrealized P&L: ${pnl:.2f}
- Net Asset Value: ${nav:.2f}
- Open Trades:
{open_trades_text if open_trades_text else "None"}"""
    tg_send_sync(message)

# ---------- Trading Decision Loop ----------
def trading_decision():
    global CURRENT_WATCHLIST
    if not CURRENT_WATCHLIST:
        CURRENT_WATCHLIST = update_watch_list()
    try:
        prices = oanda_get_prices(CURRENT_WATCHLIST)
        if not prices:
            return
        open_trades = oanda_get_open_trades()
        news = fetch_news()
        system_prompt = f"""You are an autonomous trading bot with a $100 allocation on OANDA demo.
You can trade only: {', '.join(CURRENT_WATCHLIST)}.
Respond with a JSON object:
{{"action":"BUY"|"SELL"|"HOLD", "instrument":"...", "units":1000, "stop_loss":1.0500, "take_profit":1.0600, "timeframe":"M5", "reasoning":"..."}}
If HOLD, other fields can be null. Keep total exposure ≤ $100."""
        user_prompt = f"Current time: {datetime.now().isoformat()}\nPrices: {json.dumps(prices)}\nOpen positions: {json.dumps(open_trades)}\nNews: {news}\n\nAnalyse and provide the next trade action in JSON."
        response = deepseek_chat(user_prompt, system_prompt)
        if not response:
            return
        try:
            decision = json.loads(response)
        except:
            if "```" in response:
                snippet = response.split("```")[1]
                if snippet.startswith("json"):
                    snippet = snippet[4:]
                decision = json.loads(snippet.strip())
            else:
                return
        if decision.get("action") == "HOLD":
            return
        instrument = decision.get("instrument")
        desired_units = decision.get("units", 0)
        if instrument not in CURRENT_WATCHLIST or desired_units == 0:
            return

        # Get current price for exposure calculation
        current_price = (prices[instrument]["bid"] + prices[instrument]["ask"]) / 2
        # Adjust units to instrument constraints
        valid_units = adjust_units_to_instrument(instrument, desired_units, current_price)
        if valid_units == 0:
            logger.info(f"Trade for {instrument} rejected after unit adjustment.")
            return

        # Final safety: ensure exposure ≤ $100
        exposure = abs(valid_units) * current_price
        if exposure > ALLOCATED_CAPITAL:
            logger.info(f"Trade rejected: exposure ${exposure:.2f} > $100.")
            return

        oanda_place_order(instrument, valid_units, decision.get("stop_loss"), decision.get("take_profit"))
    except Exception as e:
        logger.exception(f"Trading decision error: {e}")

# ---------- Watch List & Instrument Refresh ----------
def refresh_watch_list_job():
    global CURRENT_WATCHLIST
    CURRENT_WATCHLIST = update_watch_list()

def refresh_instrument_list_job():
    update_valid_instruments()

# ---------- Scheduler ----------
def run_scheduler():
    update_valid_instruments()

    schedule.every(5).minutes.do(mcp_optimization_runner)
    schedule.every().day.at("07:00").do(morning_analysis)
    schedule.every().day.at("21:00").do(night_performance)
    schedule.every(1).minutes.do(trading_decision)
    schedule.every(4).hours.do(refresh_watch_list_job)
    schedule.every(6).hours.do(refresh_instrument_list_job)

    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            logger.error(f"Unhandled exception in scheduler: {e}", exc_info=True)
        time.sleep(1)

# ---------- FastAPI App ----------
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global CURRENT_WATCHLIST, _STARTUP_MESSAGE_SENT
    update_valid_instruments()
    try:
        CURRENT_WATCHLIST = update_watch_list()
    except Exception as e:
        logger.error(f"Initial watch list failed: {e}")
        CURRENT_WATCHLIST = _filtered_default_watchlist()
    threading.Thread(target=run_scheduler, daemon=True).start()
    if not _STARTUP_MESSAGE_SENT:
        await send_telegram_message(
            f"🤖 Trading bot started. Allocated capital: $100.\n"
            f"Watch list: {', '.join(CURRENT_WATCHLIST)}"
        )
        _STARTUP_MESSAGE_SENT = True

@app.get("/")
@app.head("/")
def root():
    return {"message": "Trading bot is running. Use /health for status."}

@app.get("/health")
def health():
    return {"status": "ok", "best_sharpe": llm_strategy_optimizer.best_sharpe, "watch_list": CURRENT_WATCHLIST}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
