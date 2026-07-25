"""
OANDA Strategy Optimizer – DeepSeek generates complete Python trading strategies.
Unlimited strategy freedom – the AI writes the bt.Strategy subclass directly.
With timeout and resilient MCP response handling.
"""
import os, json, time, asyncio, requests, signal
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
            if "429" in str(e) or "503" in str(e):
                LAST_RATE_LIMIT = time.time()
                delay = LLM_RETRY_DELAY * (attempt + 1)
                print(f"Retrying in {delay}s…")
                time.sleep(delay)
            else:
                break
    return ""

# ---------- AI helpers ----------
def ai_pick_instruments() -> list:
    system = (
        "You are a senior financial market analyst. "
        "Select the 5 most important OANDA instruments to optimise trading strategies for today. "
        "Return ONLY a JSON array of OANDA instrument names, e.g. ['EUR_USD','XAU_USD']."
    )
    response = deepseek_chat("What are the top 5 instruments to optimise trading strategies for today?", system)
    if not response:
        return []
    try:
        if "```" in response:
            response = response.split("```")[1].replace("json", "").strip()
        instruments = json.loads(response)
        if isinstance(instruments, list):
            return instruments[:5]
    except:
        pass
    return []

def ai_generate_strategy(instrument: str, current_best: dict = None) -> str:
    system = (
        "You are an expert algorithmic trader. Write a COMPLETE Python class named 'UserStrategy' "
        "that inherits from backtrader.Strategy. The class must have at least an __init__ method "
        "and a next() method. You can use any standard backtrader indicators (bt.indicators). "
        "Include stop-loss, take-profit, or any risk management you want. "
        "Return ONLY the Python code, no markdown, no explanations.\n\n"
        "Example:\n"
        "class UserStrategy(bt.Strategy):\n"
        "    def __init__(self):\n"
        "        self.sma_fast = bt.indicators.SMA(self.data.close, period=10)\n"
        "        self.sma_slow = bt.indicators.SMA(self.data.close, period=30)\n"
        "        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)\n\n"
        "    def next(self):\n"
        "        if self.crossover > 0:\n"
        "            self.buy()\n"
        "        elif self.crossover < 0:\n"
        "            self.sell()\n"
    )
    if current_best:
        prompt = (
            f"Write a new, improved Python trading strategy for {instrument} H1. "
            f"Current best strategy (Sharpe {current_best.get('sharpe', 'N/A')}):\n"
            f"{current_best.get('code', 'None')}\n\n"
            "Please propose a different approach that might yield a higher Sharpe."
        )
    else:
        prompt = f"Write a Python trading strategy for {instrument} H1."

    return deepseek_chat(prompt, system)

# ---------- MCP client helpers with timeout ----------
SERVER_PARAMS = StdioServerParameters(
    command="python",
    args=["oanda_mcp_server.py"],
    env=os.environ.copy()
)

async def backtest(session, instrument: str, code: str) -> float:
    try:
        async with asyncio.timeout(30):
            result = await session.call_tool("backtest_python_strategy", {
                "instrument": instrument,
                "code": code
            })
    except asyncio.TimeoutError:
        print("   ⏰ MCP backtest call timed out (30s).")
        return 0.0

    if result.content and len(result.content) > 0:
        text = result.content[0].text
        print(f"   Raw backtest response: {text[:500]}")
        if not text or not text.strip():
            print("   ⚠️ Empty response from MCP server.")
            return 0.0
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            print("   ⚠️ Invalid JSON in backtest response.")
            return 0.0
        if "error" in data:
            print(f"   Backtest error: {data['error']}")
            return 0.0
        return data.get("sharpe", 0.0)
    print("   ⚠️ No content in backtest response.")
    return 0.0

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

    try:
        state = read_gist(gist_id)
        best_strategies = state.get("best_strategies", {})
    except:
        best_strategies = {}

    instruments = ai_pick_instruments()
    if not instruments:
        instruments = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD", "US30_USD"]
        print("AI selection failed, using default list.")
    print(f"AI selected instruments: {instruments}")

    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            for instrument in instruments[:MAX_INSTRUMENTS_PER_RUN]:
                current_best = best_strategies.get(instrument)
                current_sharpe = current_best.get("sharpe", -9999) if current_best else -9999
                print(f"\n📊 Optimizing {instrument} (best Sharpe: {current_sharpe:.3f})…")

                code = ai_generate_strategy(instrument, current_best)
                if not code:
                    print("   ❌ Could not get strategy code.")
                    continue
                if "```" in code:
                    code = code.split("```")[1]
                    if code.startswith("python"):
                        code = code[6:].strip()

                print(f"   Generated strategy (first 200 chars): {code[:200]}")

                sharpe = await backtest(session, instrument, code)
                print(f"   Sharpe = {sharpe:.3f}")

                if sharpe > current_sharpe:
                    best_strategies[instrument] = {
                        "code": code,
                        "sharpe": sharpe,
                        "optimized_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                    }
                    print(f"   ✅ Improved! New best Sharpe: {sharpe:.3f}")

                    write_gist(gist_id, {
                        "virtual_balance": state.get("virtual_balance", 100.0),
                        "trading_paused": state.get("trading_paused", False),
                        "best_strategies": best_strategies,
                        "last_optimized": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                    })

                await asyncio.sleep(3)

    print(f"\n🏁 Optimization finished. Strategies saved: {list(best_strategies.keys())}")

if __name__ == "__main__":
    asyncio.run(main())
