"""
OANDA Strategy Optimizer – DeepSeek generates complete Python trading strategies.
- Fetches real OANDA instruments from MCP server and lets DeepSeek choose 5 to optimise.
- Only saves strategies with positive total return.
- Uses detailed metrics for feedback.
"""
import os, json, time, asyncio, requests
from collections import deque
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---------- Config ----------
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-ai/deepseek-v4-flash")
GITHUB_GIST_TOKEN = os.environ["GITHUB_GIST_TOKEN"]
GIST_ID = os.environ.get("GIST_ID")

OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID", "")
OANDA_API_KEY = os.environ.get("OANDA_API_KEY", "")
OANDA_ENV = os.environ.get("OANDA_ENV", "practice")

MAX_INSTRUMENTS_PER_RUN = int(os.environ.get("MAX_INSTRUMENTS_PER_RUN", "1"))
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY = 5
RATE_LIMIT_COOLDOWN_SEC = 30

llm_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)

# ---------- Default fallback strategy ----------
DEFAULT_STRATEGY_CODE = """import backtrader as bt

class UserStrategy(bt.Strategy):
    params = (('sma_fast', 10), ('sma_slow', 30))

    def __init__(self):
        self.sma_fast = bt.indicators.SMA(self.data.close, period=self.p.sma_fast)
        self.sma_slow = bt.indicators.SMA(self.data.close, period=self.p.sma_slow)
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)

    def next(self):
        if self.crossover > 0:
            self.buy()
        elif self.crossover < 0:
            self.sell()
"""

# ---------- Rate limiter ----------
_llm_call_timestamps = deque()
MAX_CALLS_PER_MINUTE = 40
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_WAIT_TIMEOUT = 5
LAST_RATE_LIMIT = 0

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
        print("Rate limit reached, skipping LLM call.")
        return False
    print(f"Rate limit reached, waiting {wait_time:.1f}s…")
    time.sleep(wait_time)
    _llm_call_timestamps.popleft()
    _llm_call_timestamps.append(time.time())
    return True

def deepseek_chat(prompt: str, system: str = "") -> str:
    global LAST_RATE_LIMIT
    for attempt in range(LLM_MAX_RETRIES):
        now = time.time()
        if now - LAST_RATE_LIMIT < RATE_LIMIT_COOLDOWN_SEC:
            wait = RATE_LIMIT_COOLDOWN_SEC - (now - LAST_RATE_LIMIT)
            print(f"LLM cooldown active, waiting {wait:.0f}s…")
            time.sleep(wait)
        if not _check_rate_limit():
            return ""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = llm_client.chat.completions.create(
                model=LLM_MODEL, messages=messages,
                temperature=1, top_p=0.95, max_tokens=16384, stream=False
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"LLM API error (attempt {attempt+1}/{LLM_MAX_RETRIES}): {e}")
            if any(code in str(e) for code in ["429", "503", "529"]):
                LAST_RATE_LIMIT = time.time()
                delay = LLM_RETRY_DELAY * (attempt + 1)
                print(f"Retrying in {delay}s…")
                time.sleep(delay)
            else:
                break
    return ""

# ---------- AI helpers ----------
def ai_pick_instruments(available: list = None) -> list:
    """
    Ask DeepSeek to choose 5 instruments from the provided list.
    If no list is given or the call fails, return an empty list.
    """
    if not available:
        return []

    # Truncate the list if too long (DeepSeek can handle up to 200 tokens, this is fine)
    instruments_str = ", ".join(available[:200])
    system = (
        "You are a senior financial market analyst. "
        f"From the following list of available OANDA instruments, select the 5 most important "
        f"ones to optimise trading strategies for today. "
        "Return ONLY a valid JSON array of exactly 5 instrument names, e.g. ['EUR_USD','XAU_USD']."
    )
    prompt = f"Available instruments: {instruments_str}\n\nWhich 5 would you choose?"
    response = deepseek_chat(prompt, system)
    if not response:
        return []
    try:
        if "```" in response:
            response = response.split("```")[1].replace("json", "").strip()
        instruments = json.loads(response)
        if isinstance(instruments, list) and len(instruments) >= 1:
            # Filter to only valid instruments from the available list
            valid = [i for i in instruments if i in available]
            if len(valid) >= 1:
                return valid[:5]
    except:
        pass
    return []

def ai_generate_strategy(instrument: str, current_best: dict = None) -> str:
    system = (
        "You are an expert algorithmic trader. Write a COMPLETE Python class named 'UserStrategy' "
        "that inherits from backtrader.Strategy. The class must have at least an __init__ method "
        "and a next() method.\n\n"
        "SAFE INDICATORS (only use these):\n"
        "- bt.indicators.SMA(period=...)\n"
        "- bt.indicators.EMA(period=...)\n"
        "- bt.indicators.RSI(period=...)\n"
        "- bt.indicators.MACD(period_me1=12, period_me2=26, period_signal=9)\n"
        "- bt.indicators.CrossOver(ind1, ind2)\n"
        "- bt.indicators.BollingerBands(period=20, devfactor=2)\n"
        "- bt.indicators.ATR(period=14)\n"
        "- self.data.close, self.data.high, self.data.low\n\n"
        "RULES:\n"
        "1. Call self.buy() to enter long, self.sell() to enter short.\n"
        "2. Use self.position to check current position.\n"
        "3. The strategy MUST trade at least once per week.\n"
        "4. The goal is to end the backtest with a POSITIVE total return (>0%).\n"
        "5. If the previous strategy had a loss, try a completely different indicator or exit logic.\n"
        "6. Return ONLY the Python code, no markdown.\n\n"
        "Example:\n"
        "class UserStrategy(bt.Strategy):\n"
        "    def __init__(self):\n"
        "        self.sma_fast = bt.indicators.SMA(self.data.close, period=10)\n"
        "        self.sma_slow = bt.indicators.SMA(self.data.close, period=30)\n"
        "        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)\n"
        "    def next(self):\n"
        "        if self.crossover > 0:\n"
        "            self.buy()\n"
        "        elif self.crossover < 0:\n"
        "            self.sell()\n"
    )
    if current_best:
        prev = current_best
        prompt = (
            f"Instrument: {instrument} H1\n"
            f"Previous best strategy metrics:\n"
            f"  Total Return: {prev.get('total_return_pct', 'N/A')}%\n"
            f"  Win Rate: {prev.get('win_rate', 'N/A')}%\n"
            f"  Avg Win: {prev.get('avg_win', 'N/A')}  Avg Loss: {prev.get('avg_loss', 'N/A')}\n"
            f"  Total Trades: {prev.get('total_trades', 'N/A')}\n"
            f"Previous code:\n{prev.get('code', 'None')}\n\n"
            "Please propose a completely new strategy that will IMPROVE the total return % and win rate, "
            "and achieve a POSITIVE total return (>0%)."
        )
    else:
        prompt = f"Write a Python trading strategy for {instrument} H1 that will definitely make trades and end with a positive total return."

    code = deepseek_chat(prompt, system)
    if not code or "class UserStrategy" not in code:
        print("   ⚠️ AI returned invalid code; using default SMA crossover.")
        return DEFAULT_STRATEGY_CODE
    return code

# ---------- MCP client helpers ----------
SERVER_PARAMS = StdioServerParameters(
    command="python",
    args=["oanda_mcp_server.py"],
    env=os.environ.copy()
)

async def fetch_available_instruments(session) -> list:
    """Fetch all OANDA instruments from the MCP server."""
    try:
        result = await session.call_tool("list_instruments", {})
        if result.content and len(result.content) > 0:
            text = result.content[0].text
            data = json.loads(text)
            if isinstance(data, list):
                return data
    except Exception as e:
        print(f"   ⚠️ Failed to fetch instruments: {e}")
    return []

async def backtest(session, instrument: str, code: str) -> dict:
    default = {
        "sharpe": 0.0, "total_trades": 0, "total_return_pct": 0.0,
        "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0
    }
    try:
        async with asyncio.timeout(30):
            result = await session.call_tool("backtest_python_strategy", {
                "instrument": instrument,
                "code": code
            })
    except asyncio.TimeoutError:
        print("   ⏰ MCP backtest call timed out (30s).")
        return default
    except Exception as e:
        print(f"   ❌ MCP backtest call failed: {e}")
        return default

    if result.content and len(result.content) > 0:
        text = result.content[0].text
        print(f"   Raw backtest response: {text[:500]}")
        if not text or not text.strip():
            return default
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            print("   ⚠️ Invalid JSON in backtest response.")
            return default
        if "error" in data:
            print(f"   Backtest error: {data['error']}")
            return default
        for k in default:
            if k not in data:
                data[k] = default[k]
        return data
    return default

# ---------- Gist helpers ----------
GIST_HEADERS = {
    "Authorization": f"token {GITHUB_GIST_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def read_gist(gist_id: str) -> dict:
    resp = requests.get(f"https://api.github.com/gists/{gist_id}", headers=GIST_HEADERS)
    resp.raise_for_status()
    gist = resp.json()
    content = gist["files"].get("bot_state.json", {}).get("content", "{}")
    return json.loads(content)

def write_gist(gist_id: str, data: dict):
    payload = {"files": {"bot_state.json": {"content": json.dumps(data, indent=2)}}}
    requests.patch(f"https://api.github.com/gists/{gist_id}", headers=GIST_HEADERS, json=payload)

def create_gist(data: dict) -> str:
    payload = {
        "description": "Trading bot state",
        "public": False,
        "files": {"bot_state.json": {"content": json.dumps(data, indent=2)}}
    }
    resp = requests.post("https://api.github.com/gists", headers=GIST_HEADERS, json=payload)
    resp.raise_for_status()
    return resp.json()["id"]

# ---------- Main ----------
async def main():
    print("🚀 Starting AI‑driven universal strategy optimization…")

    gist_id = GIST_ID
    if not gist_id:
        print("No GIST_ID set – creating a new gist…")
        gist_id = create_gist({
            "virtual_balance": 100.0,
            "trading_paused": False,
            "best_strategies": {}
        })
        print(f"✅ Created gist: {gist_id}")

    state = {"virtual_balance": 100.0, "trading_paused": False, "best_strategies": {}}
    try:
        state = read_gist(gist_id)
    except Exception as e:
        print(f"⚠️ Could not read Gist ({e}). Using empty state.")
    best_strategies = state.get("best_strategies", {})

    # Default instruments in case everything fails
    DEFAULT_INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD", "US30_USD"]

    instruments = []
    try:
        async with stdio_client(SERVER_PARAMS) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # 1. Fetch real OANDA instruments
                print("📡 Fetching available OANDA instruments…")
                available = await fetch_available_instruments(session)
                print(f"   Found {len(available)} instruments.")

                # 2. Let DeepSeek pick from the real list
                if available:
                    instruments = ai_pick_instruments(available)
                if not instruments:
                    print("   AI selection failed or returned empty, using default list.")
                    instruments = DEFAULT_INSTRUMENTS
                print(f"   Selected instruments: {instruments}")

                # 3. Optimize each instrument
                for instrument in instruments[:MAX_INSTRUMENTS_PER_RUN]:
                    current_best = best_strategies.get(instrument)
                    current_return = current_best.get("total_return_pct", -9999) if current_best else -9999
                    print(f"\n📊 Optimizing {instrument} (best return: {current_return:.2f}%)…")

                    code = ai_generate_strategy(instrument, current_best)
                    if not code:
                        print("   ❌ Could not get strategy code; using default.")
                        code = DEFAULT_STRATEGY_CODE
                    if "```" in code:
                        code = code.split("```")[1]
                        if code.startswith("python"):
                            code = code[6:].strip()

                    print(f"   Generated strategy (first 200 chars): {code[:200]}")

                    result = await backtest(session, instrument, code)
                    ret = result["total_return_pct"]
                    trades = result["total_trades"]
                    sharpe = result["sharpe"]
                    win = result["win_rate"]
                    avg_win = result["avg_win"]
                    avg_loss = result["avg_loss"]
                    print(f"   Return = {ret:.2f}%, Win Rate = {win:.1f}%, "
                          f"Avg Win = {avg_win:.2f}, Avg Loss = {avg_loss:.2f}, Trades = {trades}, Sharpe = {sharpe:.3f}")

                    if ret > current_return and ret > 0:
                        best_strategies[instrument] = {
                            "code": code,
                            **result,
                            "optimized_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                        }
                        print(f"   ✅ New profitable strategy saved! Return = {ret:.2f}%")

                        write_gist(gist_id, {
                            "virtual_balance": state.get("virtual_balance", 100.0),
                            "trading_paused": state.get("trading_paused", False),
                            "best_strategies": best_strategies,
                            "last_optimized": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                        })
                    else:
                        print(f"   Skipped (not profitable or no improvement)")

                    await asyncio.sleep(3)
    except Exception as e:
        print(f"❌ Could not connect to MCP server: {e}")
        print("   Optimization skipped for this run.")

    print(f"\n🏁 Optimization finished. Strategies saved: {list(best_strategies.keys())}")

if __name__ == "__main__":
    asyncio.run(main())
