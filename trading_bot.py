"""
Autonomous Trading Bot – NVIDIA API (DeepSeek) + OANDA + Telegram + MCP (Trader.dev)
- Watch list validated against real OANDA instruments
- Trade size automatically adjusted to OANDA minimums and $100 cap
- Robust MCP error handling
- NVIDIA rate‑limit protection
"""
import os, json, time, logging, threading, asyncio
from datetime import datetime
from typing import Optional, List, Dict
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

# Load .env locally
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TradingBot")

# ---------- Config ----------
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ENV = os.getenv("OANDA_ENV", "practice")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
MCP_SERVER_COMMAND = os.getenv("MCP_SERVER_COMMAND", "python")
MCP_SERVER_ARGS = os.getenv("MCP_SERVER_ARGS", "mcp_server.py").split()
ALLOCATED_CAPITAL = 100.0

TRADERDEV_API_KEY = os.getenv("TRADERDEV_API_KEY")

# NVIDIA client
llm_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

oanda_api = API(access_token=OANDA_API_KEY, environment=OANDA_ENV)

CURRENT_WATCHLIST: List[str] = []
VALID_INSTRUMENTS: Dict[str, dict] = {}
_STARTUP_MESSAGE_SENT = False

# Rate limit guard
LAST_LLM_CALL = 0
LLM_COOLDOWN_SEC = 120  # 2 minutes

DEFAULT_INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD", "US30_USD"]

# ---------- Rate-limited LLM call ----------
def deepseek_chat(prompt: str, system: str = "") -> str:
    global LAST_LLM_CALL
    now = time.time()
    if now - LAST_LLM_CALL < LLM_COOLDOWN_SEC:
        logger.warning("LLM call skipped – rate limit cooldown active.")
        return ""
    LAST_LLM_CALL = now

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
        # If it's a rate limit, set cooldown
        if "429" in str(e):
            LAST_LLM_CALL = now + LLM_COOLDOWN_SEC
        return ""

# ---------- News ----------
def fetch_news() -> str:
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return "No news API key set."
    url = f"https://newsapi.org/v2/top-headlines?category=business&language=en&apiKey={api_key}"
    try:
        resp = requests.get(url, timeout=10)
        articles = resp.json().get("articles", [])[:5]
        return "\n".join([f"- {a['title']}" for a in articles])
    except:
        return "News fetch error."

# ---------- Instrument fetching ----------
def get_valid_instruments() -> Dict[str, dict]:
    try:
        r = accounts.AccountInstruments(accountID=OANDA_ACCOUNT_ID)
        resp = oanda_api.request(r)
        instruments = {}
        for instr in resp.get("instruments", []):
            instruments[instr["name"]] = {
                "minTradeSize": float(instr.get("minimumTradeSize", 1)),
                "maxOrderUnits": float(instr.get("maximumOrderUnits", 1000000)),
            }
        logger.info(f"Fetched {len(instruments)} valid instruments.")
        return instruments
    except Exception as e:
        logger.error(f"Instrument fetch error: {e}")
        return {}

def update_valid_instruments():
    global VALID_INSTRUMENTS
    VALID_INSTRUMENTS = get_valid_instruments()
    if not VALID_INSTRUMENTS:
        VALID_INSTRUMENTS = {i: {"minTradeSize": 1, "maxOrderUnits": 1000000} for i in DEFAULT_INSTRUMENTS}

# ---------- Watch list ----------
def update_watch_list() -> List[str]:
    if not VALID_INSTRUMENTS:
        update_valid_instruments()

    news = fetch_news()
    system = "Select 5 OANDA instruments. Return ONLY a JSON array: ['EUR_USD','XAU_USD']"
    prompt = f"News:\n{news}\nTime: {datetime.now().isoformat()}\nProvide JSON array."
    response = deepseek_chat(prompt, system)
    if not response:
        return _filtered_default_watchlist()

    try:
        if "```" in response:
            response = response.split("```")[1].replace("json", "")
        watchlist = json.loads(response.strip())
        if isinstance(watchlist, list):
            valid = [i for i in watchlist if i in VALID_INSTRUMENTS]
            if len(valid) < 3:
                return _filtered_default_watchlist()
            while len(valid) < 5:
                for d in DEFAULT_INSTRUMENTS:
                    if d not in valid and d in VALID_INSTRUMENTS:
                        valid.append(d)
                        if len(valid) >= 5:
                            break
            logger.info(f"Watch list: {valid}")
            return valid[:5]
    except:
        pass
    return _filtered_default_watchlist()

def _filtered_default_watchlist():
    return [i for i in DEFAULT_INSTRUMENTS if i in VALID_INSTRUMENTS][:5]

# ---------- OANDA helpers ----------
def oanda_get_prices(instruments):
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
    info = VALID_INSTRUMENTS.get(instrument)
    if not info:
        return 0
    min_u = int(info["minTradeSize"])
    max_u = int(info["maxOrderUnits"])
    abs_units = abs(desired_units)
    if abs_units < min_u:
        abs_units = min_u
    if abs_units > max_u:
        abs_units = max_u
    # Fit under $100
    max_allowed = int(ALLOCATED_CAPITAL / price)
    if max_allowed < min_u:
        logger.info(f"{instrument} min trade ${min_u*price:.2f} > $100, impossible.")
        return 0
    abs_units = min(abs_units, max_allowed)
    if abs_units < min_u:
        abs_units = min_u
    return abs_units if desired_units >= 0 else -abs_units

def oanda_place_order(instrument, units, sl=None, tp=None):
    data = {"order": {
        "type": "MARKET", "instrument": instrument,
        "units": str(units), "timeInForce": "FOK"
    }}
    if sl:
        data["order"]["stopLossOnFill"] = {"price": str(round(sl, 5))}
    if tp:
        data["order"]["takeProfitOnFill"] = {"price": str(round(tp, 5))}
    try:
        resp = oanda_api.request(orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=data))
        logger.info(f"Order placed: {resp}")
        return resp
    except Exception as e:
        logger.error(f"Order error: {e}")
        return None

def oanda_get_open_trades():
    resp = oanda_api.request(trades.OpenTrades(accountID=OANDA_ACCOUNT_ID))
    return resp.get("trades", [])

def oanda_get_account_summary():
    return oanda_api.request(accounts.AccountSummary(accountID=OANDA_ACCOUNT_ID))["account"]

# ---------- Telegram ----------
async def send_telegram_message(text):
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
    except Exception as e:
        logger.error(f"Telegram error: {e}")

def tg_send_sync(text):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is None:
        asyncio.run(send_telegram_message(text))
    else:
        asyncio.ensure_future(send_telegram_message(text))

# ---------- MCP Optimizer (robust) ----------
class LLMStrategyOptimizer:
    def __init__(self):
        self.best_sharpe = -9999

    async def _call_mcp_backtest(self, strategy: dict, instrument="EUR_USD", timeframe="H1") -> float:
        server_params = StdioServerParameters(command=MCP_SERVER_COMMAND, args=MCP_SERVER_ARGS)
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    if not any(t.name == "backtest_strategy" for t in tools):
                        logger.error("MCP missing backtest_strategy tool.")
                        return 0.0
                    result = await session.call_tool(
                        "backtest_strategy",
                        arguments={"strategy": strategy, "instrument": instrument, "timeframe": timeframe}
                    )
                    if result.content and len(result.content) > 0:
                        return float(result.content[0].text)
                    return 0.0
        except Exception as e:
            logger.error(f"MCP backtest failed: {e}")
            return 0.0

    async def run_optimization_loop(self):
        system = "You are a quant strategist. Output a JSON strategy for EUR/USD H1. Only JSON."
        user = "Propose an initial trading strategy."
        response = deepseek_chat(user, system)
        if not response:
            return
        try:
            strat = json.loads(response)
        except:
            return
        for i in range(5):  # reduced iterations to save API calls
            sharpe = await self._call_mcp_backtest(strat)
            logger.info(f"Iteration {i+1} Sharpe = {sharpe:.3f}")
            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                if sharpe >= 2.0:
                    logger.info("Amazing strategy found!")
                    break
            # Improve
            prompt = f"Improve this strategy (Sharpe {sharpe:.3f}): {json.dumps(strat)}"
            response = deepseek_chat(prompt, system)
            if not response:
                break
            try:
                strat = json.loads(response)
            except:
                break

llm_strategy_optimizer = LLMStrategyOptimizer()

def mcp_optimization_runner():
    try:
        asyncio.run(llm_strategy_optimizer.run_optimization_loop())
    except Exception as e:
        logger.error(f"Optimization loop error: {e}")

# ---------- Jobs ----------
def morning_analysis():
    global CURRENT_WATCHLIST
    CURRENT_WATCHLIST = update_watch_list()
    if not CURRENT_WATCHLIST:
        return tg_send_sync("🌅 Morning analysis skipped.")
    prices = oanda_get_prices(CURRENT_WATCHLIST)
    if not prices:
        return tg_send_sync("🌅 OANDA data unavailable.")
    news = fetch_news()
    prompt = f"Prices:\n{json.dumps(prices)}\nNews:\n{news}\nGive a short market outlook."
    analysis = deepseek_chat(prompt, "You are a market analyst.")
    if analysis:
        tg_send_sync(f"🌅 Morning Analysis ({datetime.now().strftime('%Y-%m-%d')})\nWatch list: {', '.join(CURRENT_WATCHLIST)}\n\n{analysis}")

def night_performance():
    try:
        summary = oanda_get_account_summary()
    except Exception as e:
        return tg_send_sync(f"🌙 Night report error: {e}")
    trades_list = oanda_get_open_trades()
    pnl = float(summary.get("unrealizedPL", 0))
    balance = float(summary.get("balance", 0))
    nav = float(summary.get("NAV", 0))
    open_text = "\n".join([f"{t['instrument']} {t['currentUnits']} @ {t['price']}" for t in trades_list])
    tg_send_sync(f"🌙 Night Report ({datetime.now().strftime('%Y-%m-%d')})\nBalance: ${balance:.2f}\nUnrealized P&L: ${pnl:.2f}\nNAV: ${nav:.2f}\nOpen:\n{open_text or 'None'}")

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
        system = f"""You are a trading bot. Trade only: {', '.join(CURRENT_WATCHLIST)}.
Return JSON: {{"action":"BUY"|"SELL"|"HOLD","instrument":"...","units":1000,"stop_loss":...,"take_profit":...,"timeframe":"M5"}}.
Exposure ≤ $100. If HOLD, other fields null."""
        user = f"Time: {datetime.now().isoformat()}\nPrices: {json.dumps(prices)}\nOpen: {json.dumps(open_trades)}\nNews: {news}\nAction?"
        response = deepseek_chat(user, system)
        if not response:
            return
        try:
            decision = json.loads(response)
        except:
            return
        if decision.get("action") == "HOLD":
            return
        instrument = decision.get("instrument")
        units = decision.get("units", 0)
        if instrument not in CURRENT_WATCHLIST or units == 0:
            return
        price = (prices[instrument]["bid"] + prices[instrument]["ask"]) / 2
        valid_units = adjust_units_to_instrument(instrument, units, price)
        if valid_units == 0:
            return
        oanda_place_order(instrument, valid_units, decision.get("stop_loss"), decision.get("take_profit"))
    except Exception as e:
        logger.exception(f"Trade error: {e}")

def refresh_watch_list_job():
    global CURRENT_WATCHLIST
    CURRENT_WATCHLIST = update_watch_list()

def refresh_instruments_job():
    update_valid_instruments()

# ---------- Scheduler ----------
def run_scheduler():
    update_valid_instruments()
    schedule.every(5).minutes.do(mcp_optimization_runner)
    schedule.every().day.at("07:00").do(morning_analysis)
    schedule.every().day.at("21:00").do(night_performance)
    schedule.every(1).minutes.do(trading_decision)
    schedule.every(4).hours.do(refresh_watch_list_job)
    schedule.every(12).hours.do(refresh_instruments_job)

    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
        time.sleep(1)

# ---------- FastAPI ----------
app = FastAPI()

@app.on_event("startup")
async def startup():
    global CURRENT_WATCHLIST, _STARTUP_MESSAGE_SENT
    update_valid_instruments()
    CURRENT_WATCHLIST = update_watch_list()
    threading.Thread(target=run_scheduler, daemon=True).start()
    if not _STARTUP_MESSAGE_SENT:
        await send_telegram_message(f"🤖 Bot started. Capital: $100.\nWatch: {', '.join(CURRENT_WATCHLIST)}")
        _STARTUP_MESSAGE_SENT = True

@app.get("/")
@app.head("/")
def root():
    return {"message": "Trading bot running."}

@app.get("/health")
def health():
    return {"status": "ok", "best_sharpe": llm_strategy_optimizer.best_sharpe, "watch_list": CURRENT_WATCHLIST}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
