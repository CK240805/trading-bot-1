"""
Autonomous Trading Bot – NVIDIA API (DeepSeek) + OANDA + Telegram + MCP (Trader.dev)
Dynamic watch list determined by LLM.
LLM-driven strategy optimization using Trader.dev via MCP proxy.
Set all required environment variables before running.
"""
import os, json, time, logging, threading, asyncio
from datetime import datetime
from typing import Optional, List
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

# ---------- NVIDIA LLM Chat ----------
def deepseek_chat(prompt: str, system: str = "") -> str:
    """Send a prompt to NVIDIA's DeepSeek model and return the text response."""
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

# ---------- Dynamic Watch List ----------
def update_watch_list() -> List[str]:
    news = fetch_news()
    system = (
        "You are a professional financial market analyst. "
        "Your task is to select the best 5 instruments to trade on the OANDA platform "
        "for the upcoming session, considering current market conditions and news. "
        "Return ONLY a valid JSON array of strings, with instrument names exactly as OANDA uses them "
        "(e.g., 'EUR_USD', 'XAU_USD', 'US30_USD'). No other text."
    )
    prompt = f"Recent financial news headlines:\n{news}\n\nCurrent datetime: {datetime.now().isoformat()}\n\nProvide the JSON array."
    response = deepseek_chat(prompt, system)
    if not response:
        logger.warning("LLM call failed, using default watch list.")
        return ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD", "US30_USD"]
    try:
        if "```" in response:
            snippet = response.split("```")[1]
            if snippet.startswith("json"):
                snippet = snippet[4:]
            watchlist = json.loads(snippet.strip())
        else:
            watchlist = json.loads(response.strip())
        if isinstance(watchlist, list) and all(isinstance(i, str) for i in watchlist):
            logger.info(f"Updated watch list: {watchlist}")
            return watchlist
    except Exception as e:
        logger.error(f"Failed to parse watch list: {e}")
    logger.warning("Using default watch list.")
    return ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD", "US30_USD"]

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

# ---------- Risk Manager ----------
def calculate_exposure(instrument: str, units: int, current_price: float) -> float:
    return abs(units) * current_price

def check_risk_allocation(instrument: str, units: int) -> bool:
    prices = oanda_get_prices([instrument])
    if instrument not in prices:
        return False
    price = (prices[instrument]["bid"] + prices[instrument]["ask"]) / 2
    new_exposure = calculate_exposure(instrument, units, price)
    trades_list = oanda_get_open_trades()
    total_current = 0.0
    for t in trades_list:
        t_instr = t["instrument"]
        t_units = int(t["currentUnits"])
        t_price = float(t["price"])
        total_current += calculate_exposure(t_instr, t_units, t_price)
    if total_current + new_exposure > ALLOCATED_CAPITAL:
        logger.info(f"Trade rejected: exposure {total_current+new_exposure:.2f} > ${ALLOCATED_CAPITAL}")
        return False
    return True

# ---------- Telegram ----------
async def send_telegram_message(text: str):
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
    except TelegramError as e:
        logger.error(f"Telegram error: {e}")

def tg_send_sync(text: str):
    """Send a Telegram message from a synchronous context (scheduler thread)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is None:
        asyncio.run(send_telegram_message(text))
    else:
        asyncio.ensure_future(send_telegram_message(text))

# ---------- LLM-Driven Strategy Optimizer (MCP proxy -> Trader.dev) ----------
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
                # MCP returns a CallToolResult with a list of content items
                if result.content and len(result.content) > 0:
                    sharpe_str = result.content[0].text
                    return float(sharpe_str)
                return 0.0

    async def run_optimization_loop(self):
        system = (
            "You are an expert quantitative trading strategist. "
            "You must output a valid JSON object that represents a complete trading strategy "
            "in the exact format required by the Trader.dev backtesting API. "
            "You are free to use any indicators, entry/exit rules, and risk management. "
            "Reply ONLY with the JSON object, no other text."
        )
        user = "Propose an initial trading strategy for EUR/USD on H1 timeframe. Be creative but use standard indicators."
        response = deepseek_chat(user, system)
        if not response:
            logger.error("LLM did not return a strategy proposal.")
            return
        try:
            current_strategy = json.loads(response)
        except:
            logger.error("Failed to parse initial strategy from LLM")
            return

        for i in range(10):
            logger.info(f"Iteration {i+1}: Testing strategy...")
            sharpe = await self._call_mcp_backtest(current_strategy)
            logger.info(f"Sharpe = {sharpe:.3f}")

            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                self.best_strategy = current_strategy
                if sharpe >= 2.0:
                    logger.info(f"Amazing strategy found! Sharpe = {sharpe:.3f}")
                    break

            improvement_prompt = (
                f"The last strategy (JSON below) achieved a Sharpe ratio of {sharpe:.3f}. "
                f"Propose an improved version of this strategy that you expect will have a higher Sharpe. "
                f"Return ONLY the new JSON strategy, no additional text.\n\n"
                f"Previous strategy: {json.dumps(current_strategy)}"
            )
            response = deepseek_chat(improvement_prompt, system)
            if not response:
                logger.error("LLM improvement call failed.")
                break
            try:
                current_strategy = json.loads(response)
            except:
                logger.error("Could not parse improved strategy, stopping loop.")
                break

        logger.info(f"Optimization ended. Best Sharpe: {self.best_sharpe:.3f}")

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
        tg_send_sync("🌅 OANDA data unavailable – check API keys or instrument names.")
        return
    news = fetch_news()
    prompt = f"""Current market snapshot for the selected instruments (bid/ask):
{json.dumps(prices, indent=2)}
Recent news headlines:
{news}

Provide a concise morning market outlook for the above instruments. Keep under 400 words."""
    analysis = deepseek_chat(prompt, "You are a professional financial market analyst.")
    if not analysis:
        tg_send_sync("🌅 Morning analysis failed – LLM unavailable.")
        return
    message = f"🌅 *Morning Market Analysis* ({datetime.now().strftime('%Y-%m-%d')})\n\n🎯 Watch list: {', '.join(CURRENT_WATCHLIST)}\n\n{analysis}"
    tg_send_sync(message)

# ---------- Night Performance Report Job ----------
def night_performance():
    try:
        summary = oanda_get_account_summary()
    except Exception as e:
        tg_send_sync(f"🌙 Night report failed – can't reach OANDA: {e}")
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
            logger.warning("No price data for watch list, skipping trading decision.")
            return
        open_trades = oanda_get_open_trades()
        news = fetch_news()
        system_prompt = f"""You are an autonomous trading bot. You manage a $100 allocation on OANDA demo.
You can trade only instruments from this list: {', '.join(CURRENT_WATCHLIST)}.
Your goal is to maximise profit while respecting a total exposure of $100.
Respond ONLY with a JSON object: {{"action":"BUY"|"SELL"|"HOLD", "instrument":"...", "units":1000, "stop_loss":1.0500, "take_profit":1.0600, "timeframe":"M5", "reasoning":"..."}}.
If HOLD, other fields can be null.
Ensure |units| * price <= $100."""
        user_prompt = f"Current time: {datetime.now().isoformat()}\nPrices: {json.dumps(prices)}\nOpen positions: {json.dumps(open_trades)}\nNews: {news}\n\nAnalyse and provide the next trade action in JSON."
        response = deepseek_chat(user_prompt, system_prompt)
        if not response:
            logger.error("LLM trading decision failed.")
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
                logger.error("Could not parse trading decision.")
                return
        if decision.get("action") == "HOLD":
            return
        instrument = decision.get("instrument")
        units = decision.get("units", 0)
        if instrument not in CURRENT_WATCHLIST or units == 0:
            logger.info(f"Invalid trade: {instrument} / {units}")
            return
        if not check_risk_allocation(instrument, units):
            return
        oanda_place_order(instrument, units, decision.get("stop_loss"), decision.get("take_profit"))
    except Exception as e:
        logger.exception(f"Trading decision error: {e}")

# ---------- Watch List Refresh ----------
def refresh_watch_list_job():
    global CURRENT_WATCHLIST
    CURRENT_WATCHLIST = update_watch_list()

# ---------- Scheduler Thread ----------
def run_scheduler():
    schedule.every(5).minutes.do(mcp_optimization_runner)
    schedule.every().day.at("07:00").do(morning_analysis)
    schedule.every().day.at("21:00").do(night_performance)
    schedule.every(1).minutes.do(trading_decision)
    schedule.every(4).hours.do(refresh_watch_list_job)

    while True:
        schedule.run_pending()
        time.sleep(1)

# ---------- FastAPI App ----------
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global CURRENT_WATCHLIST
    try:
        CURRENT_WATCHLIST = update_watch_list()
    except Exception as e:
        logger.error(f"Initial watch list failed: {e}")
        CURRENT_WATCHLIST = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD", "US30_USD"]
    threading.Thread(target=run_scheduler, daemon=True).start()
    await send_telegram_message(
        f"🤖 Trading bot started. Allocated capital: $100.\n"
        f"Watch list: {', '.join(CURRENT_WATCHLIST)}"
    )

@app.get("/health")
def health():
    return {"status": "ok", "best_sharpe": llm_strategy_optimizer.best_sharpe, "watch_list": CURRENT_WATCHLIST}

@app.get("/")
def root():
    return {"message": "Trading bot is running. Use /health for status."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
