"""
Autonomous Trading Bot – NVIDIA API (DeepSeek) + OANDA + Telegram + Trader.dev MCP
- Rate limiter: max 40 LLM calls per minute (waits or skips)
- Cooldown on 429/503 as additional back‑off
- Live news from Google News RSS, analysed by DeepSeek
- Order status correctly checked (FILL / CANCELLED / REJECTED)
- Handles MCP tools as tuples
- Live dashboard at /dashboard
"""
import os, json, time, logging, threading, asyncio
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
MCP_BACKTEST_TOOL = os.getenv("MCP_BACKTEST_TOOL")      # optional
TRADERDEV_API_KEY = os.getenv("TRADERDEV_API_KEY")      # for Trader.dev MCP auth
ALLOCATED_CAPITAL = 100.0

# NVIDIA client
llm_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

oanda_api = API(access_token=OANDA_API_KEY, environment=OANDA_ENV)

CURRENT_WATCHLIST: List[str] = []
VALID_INSTRUMENTS: Dict[str, dict] = {}
_STARTUP_MESSAGE_SENT = False

# Rate limit cooldown (triggers on 429 and 503)
LAST_RATE_LIMIT = 0
RATE_LIMIT_COOLDOWN_SEC = 120

# Rate limiter: max 40 calls per 60 seconds
_llm_call_timestamps = deque()
MAX_CALLS_PER_MINUTE = 40
RATE_LIMIT_WINDOW = 60   # seconds
RATE_LIMIT_WAIT_TIMEOUT = 5   # max seconds to wait before skipping

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

# ---------- Interaction log ----------
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

# ---------- Rate limiter helper ----------
def _check_rate_limit() -> bool:
    """
    Ensure we don't exceed MAX_CALLS_PER_MINUTE LLM requests.
    If the limit is reached, wait up to RATE_LIMIT_WAIT_TIMEOUT seconds for a slot.
    Returns True if we can proceed, False if we should skip.
    """
    global _llm_call_timestamps
    now = time.time()
    # Remove timestamps older than the window
    while _llm_call_timestamps and _llm_call_timestamps[0] < now - RATE_LIMIT_WINDOW:
        _llm_call_timestamps.popleft()

    if len(_llm_call_timestamps) < MAX_CALLS_PER_MINUTE:
        _llm_call_timestamps.append(now)
        return True

    # We're at the limit; wait until the oldest call expires
    oldest = _llm_call_timestamps[0]
    wait_time = oldest + RATE_LIMIT_WINDOW - now
    if wait_time > RATE_LIMIT_WAIT_TIMEOUT:
        logger.warning("Rate limit reached, skipping LLM call (wait too long).")
        return False

    logger.info(f"Rate limit reached, waiting {wait_time:.1f}s...")
    time.sleep(wait_time)
    # After waiting, try again (now oldest should be removed)
    _llm_call_timestamps.popleft()   # remove the oldest
    _llm_call_timestamps.append(time.time())
    return True

# ---------- LLM call ----------
def deepseek_chat(prompt: str, system: str = "", category: str = "general") -> str:
    global LAST_RATE_LIMIT
    now = time.time()
    if now - LAST_RATE_LIMIT < RATE_LIMIT_COOLDOWN_SEC:
        logger.warning("LLM call skipped – rate limit cooldown active.")
        log_interaction(category, "system", "LLM call blocked by cooldown.")
        return ""

    # Rate limiter check
    if not _check_rate_limit():
        log_interaction(category, "system", "LLM call skipped due to rate limit.")
        return ""

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    log_interaction(category, "bot", prompt)

    try:
        completion = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=1,
            top_p=0.95,
            max_tokens=16384,
            stream=False
        )
        response = completion.choices[0].message.content
        log_interaction(category, "ai", response)
        return response
    except Exception as e:
        logger.error(f"LLM API error: {e}")
        log_interaction("error", "system", f"LLM error: {e}")
        if "429" in str(e) or "503" in str(e):
            LAST_RATE_LIMIT = time.time()
        return ""

# ---------- Fetch raw headlines from Google News RSS ----------
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
            logger.warning("Google News RSS returned no headlines.")
            return ""
        raw = "\n".join(headlines[:10])
        _last_raw_headlines = raw
        _last_raw_headlines_time = now
        logger.info(f"Fetched {len(headlines[:10])} headlines from Google News.")
        return raw
    except Exception as e:
        logger.error(f"Failed to fetch Google News: {e}")
        return ""

# ---------- Get market news briefing ----------
def get_news_briefing(force: bool = False) -> str:
    global _last_news_briefing, _last_news_briefing_time
    now = time.time()

    if not force and _last_news_briefing and (now - _last_news_briefing_time) < NEWS_BRIEFING_MAX_AGE_SEC:
        return _last_news_briefing

    raw_headlines = fetch_raw_headlines()
    if not raw_headlines:
        return _last_news_briefing if _last_news_briefing else ""

    system = (
        "You are a senior financial news analyst. Based on the headlines below, provide a concise "
        "market briefing for a trader. Highlight key events, market sentiment, and potential impact on "
        "forex, commodities, and indices. Keep it under 200 words."
    )
    prompt = f"Recent business headlines:\n{raw_headlines}\n\nProvide a briefing."
    briefing = deepseek_chat(prompt, system, category="news_analysis")
    if briefing:
        _last_news_briefing = briefing
        _last_news_briefing_time = now
        return briefing
    else:
        return raw_headlines

# ---------- OANDA instruments ----------
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

    news_briefing = get_news_briefing(force=True)
    system = "Select 5 OANDA instruments. Return ONLY a JSON array: ['EUR_USD','XAU_USD']"
    prompt = f"Market briefing:\n{news_briefing}\n\nTime: {datetime.now().isoformat()}\nProvide JSON array." if news_briefing else f"Time: {datetime.now().isoformat()}\nProvide JSON array."
    response = deepseek_chat(prompt, system, category="watchlist")
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
        if "orderFillTransaction" in resp:
            logger.info(f"✅ Order FILLED: {instrument} {units} units")
            log_interaction("trade", "system", f"✅ Order FILLED: {instrument} {units} units")
            return resp
        elif "orderCancelTransaction" in resp:
            reason = resp["orderCancelTransaction"].get("reason", "unknown")
            msg = f"❌ Order CANCELLED: {instrument} {units} units. Reason: {reason}"
            logger.warning(msg)
            log_interaction("trade", "system", msg)
            tg_send_sync(msg)
        elif "orderRejectTransaction" in resp:
            reason = resp["orderRejectTransaction"].get("reason", "unknown")
            msg = f"❌ Order REJECTED: {instrument} {units} units. Reason: {reason}"
            logger.warning(msg)
            log_interaction("trade", "system", msg)
            tg_send_sync(msg)
        else:
            logger.info(f"Order response: {resp}")
            log_interaction("trade", "system", f"Unknown order response: {resp}")
        return None
    except Exception as e:
        logger.error(f"Order error: {e}")
        log_interaction("trade", "system", f"Order error: {e}")
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

# ---------- LLM-Driven Strategy Optimizer ----------
class LLMStrategyOptimizer:
    def __init__(self):
        self.best_strategy = None
        self.best_sharpe = -9999

    async def _find_backtest_tool(self, session) -> str:
        tools = await session.list_tools()
        tool_names = []
        for t in tools:
            if hasattr(t, 'name'):
                tool_names.append(t.name)
            elif isinstance(t, (tuple, list)) and len(t) > 0:
                tool_names.append(str(t[0]))
            elif isinstance(t, dict) and 'name' in t:
                tool_names.append(t['name'])
            else:
                tool_names.append(str(t))
        log_interaction("optimization", "system", f"Available MCP tools: {tool_names}")
        if MCP_BACKTEST_TOOL:
            if MCP_BACKTEST_TOOL in tool_names:
                return MCP_BACKTEST_TOOL
            else:
                log_interaction("optimization", "system", f"Specified tool '{MCP_BACKTEST_TOOL}' not found.")
                return None
        for name in tool_names:
            if "backtest" in name.lower():
                log_interaction("optimization", "system", f"Auto-selected backtest tool: {name}")
                return name
        log_interaction("optimization", "system", "No backtest tool found.")
        return None

    async def _call_mcp_backtest(self, strategy: dict, instrument="EUR_USD", timeframe="H1") -> float:
        headers = {}
        if TRADERDEV_API_KEY:
            headers["Authorization"] = f"Bearer {TRADERDEV_API_KEY}"

        try:
            async with sse_client(MCP_SERVER_URL, headers=headers) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tool_name = await self._find_backtest_tool(session)
                    if not tool_name:
                        return 0.0

                    result = await session.call_tool(
                        tool_name,
                        arguments={
                            "strategy": strategy,
                            "instrument": instrument,
                            "timeframe": timeframe
                        }
                    )
                    if result.content and len(result.content) > 0:
                        text = result.content[0].text
                        try:
                            data = json.loads(text)
                            sharpe = data.get("sharpe") or data.get("sharpe_ratio") or \
                                     data.get("performance", {}).get("sharpe") or 0.0
                            return float(sharpe)
                        except:
                            return float(text)
                    return 0.0
        except Exception as e:
            if hasattr(e, 'exceptions'):
                for sub_exc in e.exceptions:
                    logger.error(f"MCP inner exception: {sub_exc}", exc_info=True)
                    log_interaction("optimization", "system", f"MCP inner error: {sub_exc}")
            else:
                logger.error(f"MCP backtest error: {e}", exc_info=True)
                log_interaction("optimization", "system", f"MCP backtest error: {e}")
            return 0.0

    async def run_optimization_loop(self):
        system = (
            "You are a quant strategist. Output a JSON strategy object that Trader.dev can backtest. "
            "Example: {\"type\": \"sma_crossover\", \"fast_ma\": 10, \"slow_ma\": 30, \"stop_loss_pct\": 2}. "
            "Only JSON."
        )
        user = "Propose an initial trading strategy for EUR/USD H1."
        response = deepseek_chat(user, system, category="optimization")
        if not response:
            return
        try:
            current_strategy = json.loads(response)
        except:
            return

        # Only 2 iterations (1 initial, 1 improvement)
        for i in range(2):
            log_interaction("optimization", "system", f"Iteration {i+1}: Testing strategy...")
            sharpe = await self._call_mcp_backtest(current_strategy)
            log_interaction("optimization", "system", f"Iteration {i+1} Sharpe = {sharpe:.3f}")

            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                self.best_strategy = current_strategy
                if sharpe >= 2.0:
                    log_interaction("optimization", "system", "🎉 Amazing strategy found!")
                    break

            improvement_prompt = (
                f"The last strategy (Sharpe {sharpe:.3f}) was: {json.dumps(current_strategy)}. "
                "Propose an improved version. Only JSON."
            )
            response = deepseek_chat(improvement_prompt, system, category="optimization")
            if not response:
                break
            try:
                current_strategy = json.loads(response)
            except:
                break
        log_interaction("optimization", "system", f"Optimization ended. Best Sharpe: {self.best_sharpe:.3f}")

llm_strategy_optimizer = LLMStrategyOptimizer()

def mcp_optimization_runner():
    try:
        asyncio.run(llm_strategy_optimizer.run_optimization_loop())
    except Exception as e:
        logger.error(f"Optimization loop error: {e}")

# ---------- Scheduled Jobs ----------
def morning_analysis():
    global CURRENT_WATCHLIST
    CURRENT_WATCHLIST = update_watch_list()
    if not CURRENT_WATCHLIST:
        return tg_send_sync("🌅 Morning analysis skipped.")
    prices = oanda_get_prices(CURRENT_WATCHLIST)
    if not prices:
        return tg_send_sync("🌅 OANDA data unavailable.")
    news_briefing = get_news_briefing(force=True)
    news_block = f"\nMarket news briefing:\n{news_briefing}" if news_briefing else ""
    prompt = f"Prices:\n{json.dumps(prices)}{news_block}\nGive a short market outlook."
    analysis = deepseek_chat(prompt, "You are a market analyst.", category="morning")
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
        news_briefing = get_news_briefing()
        news_block = f"\nMarket news briefing:\n{news_briefing}" if news_briefing else ""
        system = f"""You are a trading bot. Trade only: {', '.join(CURRENT_WATCHLIST)}.
Return JSON: {{"action":"BUY"|"SELL"|"HOLD","instrument":"...","units":1000,"stop_loss":...,"take_profit":...,"timeframe":"M5"}}.
Exposure ≤ $100. If HOLD, other fields null."""
        user = f"Time: {datetime.now().isoformat()}\nPrices: {json.dumps(prices)}\nOpen: {json.dumps(open_trades)}{news_block}\nAction?"
        response = deepseek_chat(user, system, category="trade")
        if not response:
            return
        try:
            decision = json.loads(response)
        except:
            log_interaction("trade", "system", "Failed to parse trade decision JSON.")
            return
        if decision.get("action") == "HOLD":
            log_interaction("trade", "system", "AI decided HOLD.")
            return
        instrument = decision.get("instrument")
        units = decision.get("units", 0)
        if instrument not in CURRENT_WATCHLIST or units == 0:
            log_interaction("trade", "system", f"Invalid trade: {instrument} / {units}")
            return
        price = (prices[instrument]["bid"] + prices[instrument]["ask"]) / 2
        valid_units = adjust_units_to_instrument(instrument, units, price)
        if valid_units == 0:
            log_interaction("trade", "system", f"Trade adjusted to 0 units, rejected.")
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
    schedule.every(1).minutes.do(trading_decision)   # 1 call/min → 60/hr, well within 40/min
    schedule.every(4).hours.do(refresh_watch_list_job)
    schedule.every(12).hours.do(refresh_instruments_job)

    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
        time.sleep(1)

# ---------- FastAPI & Dashboard ----------
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
    return {"message": "Trading bot running. Visit /dashboard for live interactions."}

@app.get("/health")
def health():
    return {"status": "ok", "best_sharpe": llm_strategy_optimizer.best_sharpe, "watch_list": CURRENT_WATCHLIST}

@app.get("/api/logs")
def api_logs():
    return list(INTERACTION_LOG)

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trading Bot Live</title>
        <meta charset="utf-8">
        <meta http-equiv="refresh" content="30">
        <style>
            body { font-family: monospace; background: #0a0a0a; color: #ccc; padding: 20px; }
            h1 { color: #00ff88; }
            .log { max-height: 80vh; overflow-y: auto; border: 1px solid #333; padding: 10px; background: #111; }
            .entry { margin-bottom: 8px; border-bottom: 1px solid #222; padding-bottom: 5px; }
            .time { color: #888; font-size: 0.85em; }
            .role { font-weight: bold; }
            .bot { color: #4da6ff; }
            .ai { color: #ffaa00; }
            .system { color: #888; }
            .optimization { color: #af5fff; }
            .watchlist { color: #5fd7ff; }
            .trade { color: #ff6666; }
            .error { color: #ff0000; }
            pre { white-space: pre-wrap; margin: 4px 0; }
        </style>
    </head>
    <body>
        <h1>🤖 Trading Bot Live</h1>
        <p>Auto‑refreshes every 30 seconds. Best Sharpe: <span id="sharpe">-</span></p>
        <div class="log" id="log"></div>
        <script>
            async function load() {
                const resp = await fetch('/api/logs');
                const logs = await resp.json();
                const logDiv = document.getElementById('log');
                logDiv.innerHTML = logs.map(e => `
                    <div class="entry">
                        <span class="time">${e.time}</span>
                        <span class="role ${e.role}">[${e.category}] ${e.role}:</span>
                        <pre>${e.content}</pre>
                    </div>`).join('');
                const sharpeResp = await fetch('/health');
                const health = await sharpeResp.json();
                document.getElementById('sharpe').innerText = health.best_sharpe.toFixed(2);
            }
            load();
            setInterval(load, 10000);
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
