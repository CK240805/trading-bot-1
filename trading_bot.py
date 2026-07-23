"""
Autonomous Trading Bot – NVIDIA API (DeepSeek) + OANDA + Telegram + Trader.dev MCP
- LLM calls retry up to 5 times on 429/503 errors
- Passes symbol explicitly to run_backtest (prevents defaulting to BTCUSDT)
- On startup, fills optimization queue with watchlist instruments
- Optimizes strategies per instrument in the background every 5 minutes
- Virtual budget ($100) with risk‑based position sizing (2% per trade)
- Saves state (budget, per‑instrument strategies) to GitHub Gist
- Stops trading if budget reaches $0
- Rate limiter: max 40 LLM calls per minute
- Live dashboard at /dashboard
"""
import os, json, time, logging, threading, asyncio, re
from datetime import datetime
from typing import Optional, List, Dict
from collections import deque
import requests
import schedule
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
from telegram import Bot
from telegram.error import TelegramError

# OANDA
from oandapyV20 import API
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.accounts as accounts

# MCP client (SSE transport)
from mcp import ClientSession
from mcp.client.sse import sse_client

# NVIDIA LLM client
from openai import OpenAI

# For Google News RSS
import xml.etree.ElementTree as ET

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
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-ai/deepseek-v4-flash")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "https://mcp.trader.dev/sse")
TRADERDEV_API_KEY = os.getenv("TRADERDEV_API_KEY")

# ---------- Gist state persistence ----------
GIST_ID = os.getenv("GIST_ID")
GITHUB_GIST_TOKEN = os.getenv("GITHUB_GIST_TOKEN")
GIST_HEADERS = {}
if GITHUB_GIST_TOKEN:
    GIST_HEADERS = {
        "Authorization": f"token {GITHUB_GIST_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

VIRTUAL_BALANCE = 100.0
TRADING_PAUSED = False
BEST_STRATEGIES: Dict[str, dict] = {}
OPTIMIZATION_QUEUE: List[str] = []
CURRENT_OPTIMIZING: Optional[str] = None

# ---------- NVIDIA client ----------
llm_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

oanda_api = API(access_token=OANDA_API_KEY, environment=OANDA_ENV)

CURRENT_WATCHLIST: List[str] = []
VALID_INSTRUMENTS: Dict[str, dict] = {}
_STARTUP_MESSAGE_SENT = False

# ---------- Rate limiter & retry ----------
_llm_call_timestamps = deque()
MAX_CALLS_PER_MINUTE = 40
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_WAIT_TIMEOUT = 5
LAST_RATE_LIMIT = 0
RATE_LIMIT_COOLDOWN_SEC = 120
LLM_MAX_RETRIES = 5
LLM_RETRY_DELAY = 10  # seconds base delay for exponential backoff

DEFAULT_INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD", "US30_USD"]

# ---------- News cache ----------
_last_raw_headlines: str = ""
_last_raw_headlines_time: float = 0.0
RAW_NEWS_MAX_AGE_SEC = 300

_last_news_briefing: str = ""
_last_news_briefing_time: float = 0.0
NEWS_BRIEFING_MAX_AGE_SEC = 1800

GOOGLE_NEWS_RSS_URL = (
    "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB"
    "?hl=en-US&gl=US&ceid=US:en"
)

INTERACTION_LOG = deque(maxlen=200)

def log_interaction(category: str, role: str, content: str):
    entry = {
        "time": datetime.utcnow().isoformat() + "Z",
        "category": category,
        "role": role,
        "content": content
    }
    INTERACTION_LOG.append(entry)
    logger.info(f"[{category}] {role}: {content[:200]}")

# ---------- Gist state persistence ----------
def load_state():
    global VIRTUAL_BALANCE, TRADING_PAUSED, BEST_STRATEGIES
    if not GIST_ID or not GITHUB_GIST_TOKEN:
        logger.warning("Gist credentials not set – using defaults.")
        return
    try:
        resp = requests.get(f"https://api.github.com/gists/{GIST_ID}", headers=GIST_HEADERS)
        resp.raise_for_status()
        gist = resp.json()
        content = gist["files"].get("bot_state.json", {}).get("content", "{}")
        state = json.loads(content)
        VIRTUAL_BALANCE = state.get("virtual_balance", 100.0)
        TRADING_PAUSED = state.get("trading_paused", False)
        BEST_STRATEGIES = state.get("best_strategies", {})
        logger.info(f"Loaded state: balance=${VIRTUAL_BALANCE:.2f}, strategies={list(BEST_STRATEGIES.keys())}")
    except Exception as e:
        logger.error(f"Failed to load state from Gist: {e}")

def save_state():
    if not GIST_ID or not GITHUB_GIST_TOKEN:
        return
    state = {
        "virtual_balance": VIRTUAL_BALANCE,
        "trading_paused": TRADING_PAUSED,
        "best_strategies": BEST_STRATEGIES,
        "last_optimized": time.strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    payload = {"files": {"bot_state.json": {"content": json.dumps(state, indent=2)}}}
    try:
        requests.patch(f"https://api.github.com/gists/{GIST_ID}", headers=GIST_HEADERS, json=payload)
    except Exception as e:
        logger.error(f"Failed to save state to Gist: {e}")

# ---------- Rate limiter ----------
def _check_rate_limit() -> bool:
    global _llm_call_timestamps
    now = time.time()
    while _llm_call_timestamps and _llm_call_timestamps[0] < now - RATE_LIMIT_WINDOW:
        _llm_call_timestamps.popleft()
    if len(_llm_call_timestamps) < MAX_CALLS_PER_MINUTE:
        _llm_call_timestamps.append(now)
        return True
    oldest = _llm_call_timestamps[0]
    wait_time = oldest + RATE_LIMIT_WINDOW - now
    if wait_time > RATE_LIMIT_WAIT_TIMEOUT:
        return False
    time.sleep(wait_time)
    _llm_call_timestamps.popleft()
    _llm_call_timestamps.append(time.time())
    return True

# ---------- LLM call with retry ----------
def deepseek_chat(prompt: str, system: str = "", category: str = "general") -> str:
    """Call LLM with exponential backoff on 429/503 errors."""
    global LAST_RATE_LIMIT
    for attempt in range(LLM_MAX_RETRIES):
        now = time.time()
        if now - LAST_RATE_LIMIT < RATE_LIMIT_COOLDOWN_SEC:
            wait = RATE_LIMIT_COOLDOWN_SEC - (now - LAST_RATE_LIMIT)
            logger.warning(f"LLM cooldown active, waiting {wait:.0f}s…")
            time.sleep(wait)

        if not _check_rate_limit():
            return ""

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        log_interaction(category, "bot", prompt)
        try:
            completion = llm_client.chat.completions.create(
                model=LLM_MODEL, messages=messages,
                temperature=1, top_p=0.95, max_tokens=16384, stream=False
            )
            response = completion.choices[0].message.content
            log_interaction(category, "ai", response)
            return response
        except Exception as e:
            logger.error(f"LLM API error (attempt {attempt+1}/{LLM_MAX_RETRIES}): {e}")
            if "429" in str(e) or "503" in str(e):
                LAST_RATE_LIMIT = time.time()
                delay = LLM_RETRY_DELAY * (attempt + 1)
                logger.info(f"Retrying in {delay}s…")
                time.sleep(delay)
            else:
                break  # non‑retryable error
    return ""

# ---------- Google News ----------
def fetch_raw_headlines() -> str:
    global _last_raw_headlines, _last_raw_headlines_time
    now = time.time()
    if _last_raw_headlines and (now - _last_raw_headlines_time) < RAW_NEWS_MAX_AGE_SEC:
        return _last_raw_headlines
    try:
        resp = requests.get(GOOGLE_NEWS_RSS_URL, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        headlines = []
        for item in root.iter("item"):
            title_elem = item.find("title")
            if title_elem is not None and title_elem.text:
                headlines.append(f"- {title_elem.text.strip()}")
        if not headlines:
            return ""
        raw = "\n".join(headlines[:10])
        _last_raw_headlines = raw
        _last_raw_headlines_time = now
        return raw
    except:
        return ""

def get_news_briefing(force: bool = False) -> str:
    global _last_news_briefing, _last_news_briefing_time
    now = time.time()
    if not force and _last_news_briefing and (now - _last_news_briefing_time) < NEWS_BRIEFING_MAX_AGE_SEC:
        return _last_news_briefing
    raw_headlines = fetch_raw_headlines()
    if not raw_headlines:
        return _last_news_briefing if _last_news_briefing else ""
    system = "Summarise these headlines into a concise briefing for a trader (under 200 words)."
    prompt = f"Headlines:\n{raw_headlines}\n\nProvide briefing."
    briefing = deepseek_chat(prompt, system, category="news_analysis")
    if briefing:
        _last_news_briefing = briefing
        _last_news_briefing_time = now
        return briefing
    return raw_headlines

# ---------- OANDA ----------
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
        return instruments
    except:
        return {}

def update_valid_instruments():
    global VALID_INSTRUMENTS
    VALID_INSTRUMENTS = get_valid_instruments()
    if not VALID_INSTRUMENTS:
        VALID_INSTRUMENTS = {i: {"minTradeSize": 1, "maxOrderUnits": 1000000} for i in DEFAULT_INSTRUMENTS}

def update_watch_list() -> List[str]:
    if not VALID_INSTRUMENTS:
        update_valid_instruments()
    news_briefing = get_news_briefing(force=True)
    system = "Select 5 OANDA instruments. Return ONLY a JSON array."
    prompt = f"Market briefing:\n{news_briefing}\nTime: {datetime.now().isoformat()}\nProvide JSON array." if news_briefing else f"Time: {datetime.now().isoformat()}\nProvide JSON array."
    response = deepseek_chat(prompt, system, category="watchlist")
    if not response:
        return [i for i in DEFAULT_INSTRUMENTS if i in VALID_INSTRUMENTS][:5]
    try:
        if "```" in response:
            response = response.split("```")[1].replace("json", "")
        watchlist = json.loads(response.strip())
        if isinstance(watchlist, list):
            valid = [i for i in watchlist if i in VALID_INSTRUMENTS]
            return valid[:5] if valid else [i for i in DEFAULT_INSTRUMENTS if i in VALID_INSTRUMENTS][:5]
    except:
        pass
    return [i for i in DEFAULT_INSTRUMENTS if i in VALID_INSTRUMENTS][:5]

def oanda_get_prices(instruments):
    params = {"instruments": ",".join(instruments)}
    r = pricing.PricingInfo(accountID=OANDA_ACCOUNT_ID, params=params)
    resp = oanda_api.request(r)
    prices = {}
    for p in resp.get("prices", []):
        if p["type"] == "PRICE":
            prices[p["instrument"]] = {"bid": float(p["bids"][0]["price"]), "ask": float(p["asks"][0]["price"])}
    return prices

def calculate_position_size(instrument, price):
    risk_amount = VIRTUAL_BALANCE * 0.02
    units = int(risk_amount / price)
    info = VALID_INSTRUMENTS.get(instrument)
    if not info:
        return 0
    min_u = int(info["minTradeSize"])
    max_u = int(info["maxOrderUnits"])
    abs_units = abs(units)
    if abs_units < min_u:
        abs_units = min_u
    if abs_units > max_u:
        abs_units = max_u
    if abs_units * price > VIRTUAL_BALANCE:
        abs_units = int(VIRTUAL_BALANCE / price)
        if abs_units < min_u:
            return 0
    return abs_units

def validate_sl_tp(direction, entry, sl, tp):
    BUFFER = 0.0001
    if direction == "BUY":
        if sl and sl >= entry - BUFFER:
            sl = None
        if tp and tp <= entry + BUFFER:
            tp = None
    else:
        if sl and sl <= entry + BUFFER:
            sl = None
        if tp and tp >= entry - BUFFER:
            tp = None
    return sl, tp

def oanda_place_order(instrument, units, sl=None, tp=None):
    direction = "BUY" if units > 0 else "SELL"
    prices = oanda_get_prices([instrument])
    entry = (prices[instrument]["bid"] + prices[instrument]["ask"]) / 2 if instrument in prices else 0.0
    sl, tp = validate_sl_tp(direction, entry, sl, tp)
    data = {"order": {"type": "MARKET", "instrument": instrument, "units": str(units), "timeInForce": "FOK"}}
    if sl:
        data["order"]["stopLossOnFill"] = {"price": str(round(sl, 5))}
    if tp:
        data["order"]["takeProfitOnFill"] = {"price": str(round(tp, 5))}
    try:
        resp = oanda_api.request(orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=data))
        if "orderFillTransaction" in resp:
            log_interaction("trade", "system", f"✅ Order FILLED: {instrument} {units}")
            return resp
        elif "orderCancelTransaction" in resp:
            reason = resp["orderCancelTransaction"].get("reason", "unknown")
            log_interaction("trade", "system", f"❌ CANCELLED: {reason}")
        elif "orderRejectTransaction" in resp:
            reason = resp["orderRejectTransaction"].get("reason", "unknown")
            log_interaction("trade", "system", f"❌ REJECTED: {reason}")
    except Exception as e:
        log_interaction("trade", "system", f"Order error: {e}")
    return None

def oanda_get_open_trades():
    return oanda_api.request(trades.OpenTrades(accountID=OANDA_ACCOUNT_ID)).get("trades", [])

def oanda_get_account_summary():
    return oanda_api.request(accounts.AccountSummary(accountID=OANDA_ACCOUNT_ID))["account"]

def update_virtual_balance():
    global VIRTUAL_BALANCE, TRADING_PAUSED
    closed = oanda_api.request(trades.TradesList(accountID=OANDA_ACCOUNT_ID, params={"state": "CLOSED", "count": 20})).get("trades", [])
    total_pnl = sum(float(t.get("realizedPL", 0)) for t in closed)
    VIRTUAL_BALANCE = 100.0 + total_pnl
    if VIRTUAL_BALANCE <= 0:
        VIRTUAL_BALANCE = 0.0
        TRADING_PAUSED = True
        tg_send_sync("🚫 Virtual budget depleted ($0). Trading paused.")
    save_state()

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

# ---------- Pine Script helpers ----------
def clean_pine_code(code: str) -> str:
    if "```" in code:
        parts = code.split("```")
        for part in parts:
            part = part.strip()
            if part.lower().startswith("pine"):
                part = part[4:].strip()
            if part.startswith("//@version"):
                code = part
                break
    code = re.sub(r'//@version\s*=\s*\d+', '//@version=6', code, count=1)
    if not code.strip().startswith('//@version'):
        code = '//@version=6\n' + code
    code = re.sub(r'[^\x00-\x7F]+', '', code)
    code = re.sub(r'pyramiding\s*=\s*\d+', 'pyramiding=1', code)
    code = re.sub(r'default_qty_type\s*=\s*strategy\.\w+', 'default_qty_type=strategy.percent_of_equity', code)
    code = re.sub(r'default_qty_value\s*=\s*\d+', 'default_qty_value=100', code)
    return code.strip()

def extract_sharpe(obj, depth=0):
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except:
            match = re.search(r'sharpe["\']?\s*[:=]\s*([0-9.]+)', obj, re.IGNORECASE)
            if match:
                return float(match.group(1))
            return None
    if depth > 10 or obj is None:
        return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "sharpe" in k.lower() and isinstance(v, (int, float)):
                return float(v)
            r = extract_sharpe(v, depth + 1)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for item in obj:
            r = extract_sharpe(item, depth + 1)
            if r is not None:
                return r
    return None

def oanda_to_traderdev_symbol(oanda_symbol: str) -> str:
    return oanda_symbol.replace("_", "")

# ---------- Per‑instrument Optimizer ----------
class LLMStrategyOptimizer:
    def __init__(self):
        pass

    async def _create_and_backtest(self, session, pine_code: str, symbol: str, timeframe: str = "1h") -> float:
        pine_code = clean_pine_code(pine_code)
        name = f"live-{symbol}-{int(time.time())}"
        result = await session.call_tool(
            "create_strategy",
            arguments={"name": name, "symbol": symbol, "timeframe": timeframe, "pineSource": pine_code}
        )
        if not result.content:
            return 0.0
        text = result.content[0].text
        if "error" in text.lower():
            return 0.0
        try:
            data = json.loads(text)
        except Exception:
            logger.warning(f"create_strategy returned non‑JSON: {text[:200]}")
            return 0.0
        sid = data.get("id")
        if not sid:
            return 0.0
        # Pass symbol explicitly to override default
        result = await session.call_tool(
            "run_backtest",
            arguments={"strategyId": sid, "symbol": symbol, "timeframe": timeframe}
        )
        if not result.content:
            return 0.0
        text = result.content[0].text
        if "error" in text.lower():
            return 0.0
        sharpe = extract_sharpe(text)
        return sharpe if sharpe is not None else 0.0

    async def optimize_for_instrument(self, oanda_instrument: str):
        global BEST_STRATEGIES, CURRENT_OPTIMIZING
        CURRENT_OPTIMIZING = oanda_instrument
        td_symbol = oanda_to_traderdev_symbol(oanda_instrument)
        log_interaction("optimization", "system", f"Optimizing {oanda_instrument} ({td_symbol})…")

        headers = {}
        if TRADERDEV_API_KEY:
            headers["Authorization"] = f"Bearer {TRADERDEV_API_KEY}"

        try:
            async with sse_client(MCP_SERVER_URL, headers=headers) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    system = (
                        f"Write a Pine Script v6 strategy for {td_symbol} on 1h. "
                        "Only use: ta.sma, ta.ema, ta.rsi, ta.macd, ta.crossover, ta.crossunder, "
                        "ta.highest, ta.lowest, ta.atr, ta.bb. //@version=6. "
                        "Use strategy() with pyramiding=1, default_qty_type=strategy.percent_of_equity, "
                        "default_qty_value=100. Output ONLY code."
                    )
                    user = f"Write a Pine Script v6 strategy for {td_symbol} 1h."
                    response = deepseek_chat(user, system, category="optimization")
                    if not response:
                        CURRENT_OPTIMIZING = None
                        return
                    current_pine = clean_pine_code(response)
                    sharpe = await self._create_and_backtest(session, current_pine, td_symbol)
                    log_interaction("optimization", "system", f"{oanda_instrument} Sharpe = {sharpe:.3f}")

                    current_best = BEST_STRATEGIES.get(oanda_instrument, {})
                    current_best_sharpe = current_best.get("sharpe", -9999)
                    if sharpe > current_best_sharpe:
                        BEST_STRATEGIES[oanda_instrument] = {
                            "pine": current_pine,
                            "sharpe": sharpe,
                            "symbol": td_symbol,
                            "optimized_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                        }
                        save_state()
                        log_interaction("optimization", "system",
                                        f"✅ New best for {oanda_instrument}: Sharpe = {sharpe:.3f}")
        except Exception as e:
            logger.error(f"Optimization error for {oanda_instrument}: {e}")
        finally:
            CURRENT_OPTIMIZING = None

llm_strategy_optimizer = LLMStrategyOptimizer()

def mcp_optimization_runner():
    global OPTIMIZATION_QUEUE, CURRENT_OPTIMIZING
    if CURRENT_OPTIMIZING is not None:
        return
    if not OPTIMIZATION_QUEUE:
        OPTIMIZATION_QUEUE = list(CURRENT_WATCHLIST)
        if not OPTIMIZATION_QUEUE:
            return
    instrument = OPTIMIZATION_QUEUE.pop(0)
    try:
        asyncio.run(llm_strategy_optimizer.optimize_for_instrument(instrument))
    except Exception as e:
        logger.error(f"Optimization runner error: {e}")

# ---------- Scheduled Jobs ----------
def morning_analysis():
    global CURRENT_WATCHLIST
    CURRENT_WATCHLIST = update_watch_list()
    if not CURRENT_WATCHLIST:
        return
    prices = oanda_get_prices(CURRENT_WATCHLIST)
    if not prices:
        return
    news_briefing = get_news_briefing(force=True)
    news_block = f"\nMarket briefing:\n{news_briefing}" if news_briefing else ""
    prompt = f"Prices:\n{json.dumps(prices)}{news_block}\nGive a short market outlook."
    analysis = deepseek_chat(prompt, "You are a market analyst.", category="morning")
    if analysis:
        tg_send_sync(f"🌅 Morning Analysis\nWatch: {', '.join(CURRENT_WATCHLIST)}\n\n{analysis}")

def night_performance():
    try:
        summary = oanda_get_account_summary()
    except:
        return
    update_virtual_balance()
    trades_list = oanda_get_open_trades()
    pnl = float(summary.get("unrealizedPL", 0))
    balance = float(summary.get("balance", 0))
    open_text = "\n".join([f"{t['instrument']} {t['currentUnits']} @ {t['price']}" for t in trades_list])
    strat_summary = "\n".join([f"{k}: Sharpe={v.get('sharpe', 'N/A'):.3f}" for k, v in BEST_STRATEGIES.items()]) if BEST_STRATEGIES else "None"
    msg = (
        f"🌙 Night Report\n"
        f"Virtual Budget: ${VIRTUAL_BALANCE:.2f}\n"
        f"Real Balance: ${balance:.2f}\n"
        f"Unrealized P&L: ${pnl:.2f}\n"
        f"Best Strategies:\n{strat_summary}\n"
        f"Open:\n{open_text or 'None'}"
    )
    tg_send_sync(msg)

def trading_decision():
    global CURRENT_WATCHLIST
    if TRADING_PAUSED:
        return
    if not CURRENT_WATCHLIST:
        CURRENT_WATCHLIST = update_watch_list()
    try:
        prices = oanda_get_prices(CURRENT_WATCHLIST)
        if not prices:
            return
        open_trades = oanda_get_open_trades()
        news_briefing = get_news_briefing()
        news_block = f"\nMarket briefing:\n{news_briefing}" if news_briefing else ""
        system = (
            f"You are a trading bot. Trade only: {', '.join(CURRENT_WATCHLIST)}. "
            "Return JSON: action (BUY/SELL/HOLD), instrument, units (negative for SELL), stop_loss, take_profit. "
            f"Virtual budget: ${VIRTUAL_BALANCE:.2f}."
        )
        user = f"Time: {datetime.now().isoformat()}\nPrices: {json.dumps(prices)}\nOpen: {json.dumps(open_trades)}{news_block}\nAction?"
        response = deepseek_chat(user, system, category="trade")
        if not response:
            return
        try:
            decision = json.loads(response)
        except:
            return
        if decision.get("action") == "HOLD":
            return
        action = decision.get("action")
        instrument = decision.get("instrument")
        units = decision.get("units", 0)
        if instrument not in CURRENT_WATCHLIST or units == 0:
            return
        if action == "SELL" and units > 0:
            units = -abs(units)
        elif action == "BUY" and units < 0:
            units = abs(units)
        price = (prices[instrument]["bid"] + prices[instrument]["ask"]) / 2
        max_units = calculate_position_size(instrument, price)
        if max_units == 0:
            return
        if abs(units) > max_units:
            units = max_units if units > 0 else -max_units
        oanda_place_order(instrument, units, decision.get("stop_loss"), decision.get("take_profit"))
    except Exception as e:
        logger.exception(f"Trade error: {e}")

def refresh_watch_list_job():
    global CURRENT_WATCHLIST, OPTIMIZATION_QUEUE
    CURRENT_WATCHLIST = update_watch_list()
    OPTIMIZATION_QUEUE = list(CURRENT_WATCHLIST)

def refresh_instruments_job():
    update_valid_instruments()

def update_balance_job():
    update_virtual_balance()

# ---------- Scheduler ----------
def run_scheduler():
    load_state()
    update_valid_instruments()
    global CURRENT_WATCHLIST, OPTIMIZATION_QUEUE
    CURRENT_WATCHLIST = update_watch_list()
    OPTIMIZATION_QUEUE = list(CURRENT_WATCHLIST)

    schedule.every(5).minutes.do(mcp_optimization_runner)
    schedule.every().day.at("07:00").do(morning_analysis)
    schedule.every().day.at("21:00").do(night_performance)
    schedule.every(1).minutes.do(trading_decision)
    schedule.every(4).hours.do(refresh_watch_list_job)
    schedule.every(12).hours.do(refresh_instruments_job)
    schedule.every(15).minutes.do(update_balance_job)

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
    global CURRENT_WATCHLIST, OPTIMIZATION_QUEUE, _STARTUP_MESSAGE_SENT
    load_state()
    update_valid_instruments()
    CURRENT_WATCHLIST = update_watch_list()
    OPTIMIZATION_QUEUE = list(CURRENT_WATCHLIST)
    threading.Thread(target=run_scheduler, daemon=True).start()
    if not _STARTUP_MESSAGE_SENT:
        strategies_count = len(BEST_STRATEGIES)
        await send_telegram_message(
            f"🤖 Bot started. Budget: ${VIRTUAL_BALANCE:.2f}. "
            f"Strategies: {strategies_count} optimized. Watch: {', '.join(CURRENT_WATCHLIST)}"
        )
        _STARTUP_MESSAGE_SENT = True

@app.get("/")
@app.head("/")
def root():
    return {"message": "Trading bot running."}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "virtual_balance": VIRTUAL_BALANCE,
        "trading_paused": TRADING_PAUSED,
        "strategies": {k: {"sharpe": v.get("sharpe")} for k, v in BEST_STRATEGIES.items()},
        "current_optimizing": CURRENT_OPTIMIZING,
        "watch_list": CURRENT_WATCHLIST
    }

@app.get("/api/logs")
def api_logs():
    return list(INTERACTION_LOG)

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """
    <!DOCTYPE html><html><head><title>Trading Bot Live</title><meta charset="utf-8"><meta http-equiv="refresh" content="30">
    <style>body{font-family:monospace;background:#0a0a0a;color:#ccc;padding:20px}h1{color:#00ff88}
    .log{max-height:80vh;overflow-y:auto;border:1px solid #333;padding:10px;background:#111}
    .entry{margin-bottom:8px;border-bottom:1px solid #222;padding-bottom:5px}
    .time{color:#888;font-size:.85em}.role{font-weight:bold}.bot{color:#4da6ff}.ai{color:#ffaa00}
    .system{color:#888}.optimization{color:#af5fff}.trade{color:#ff6666}.error{color:#ff0000}
    pre{white-space:pre-wrap;margin:4px 0}</style></head><body>
    <h1>🤖 Trading Bot Live</h1><p>Budget: $<span id="bal">0</span> | Optimizing: <span id="opt">-</span></p>
    <div id="strats"></div>
    <div class="log" id="log"></div><script>async function load(){
    const r=await fetch('/api/logs');const logs=await r.json();
    document.getElementById('log').innerHTML=logs.map(e=>`<div class="entry"><span class="time">${e.time}</span>
    <span class="role ${e.role}">[${e.category}] ${e.role}:</span><pre>${e.content}</pre></div>`).join('');
    const h=await fetch('/health');const health=await h.json();
    document.getElementById('bal').innerText=health.virtual_balance.toFixed(2);
    document.getElementById('opt').innerText=health.current_optimizing||'idle';
    const strats=health.strategies||{};
    let s='<p>Best Strategies:</p><ul>';
    for(const[k,v]of Object.entries(strats)){s+=`<li>${k}: Sharpe=${v.sharpe?.toFixed(3)||'N/A'}</li>`;}
    s+='</ul>';document.getElementById('strats').innerHTML=s;
    }load();setInterval(load,10000);</script></body></html>"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
