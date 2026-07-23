"""
Pre‑trading strategy optimization – DeepSeek + Trader.dev MCP
- Retries LLM calls on 429/503 errors
- Passes symbol explicitly to run_backtest
- Saves per‑instrument best strategies to a GitHub Gist
"""
import os, json, time, asyncio, requests, re
from collections import deque
from openai import OpenAI
from mcp import ClientSession
from mcp.client.sse import sse_client

# ---------- Config ----------
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-ai/deepseek-v4-flash")
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "https://mcp.trader.dev/sse")
TRADERDEV_API_KEY = os.environ.get("TRADERDEV_API_KEY", "")
GITHUB_GIST_TOKEN = os.environ["GITHUB_GIST_TOKEN"]
GIST_ID = os.environ.get("GIST_ID")

TIMEFRAME = "1h"
LLM_MAX_RETRIES = 5
LLM_RETRY_DELAY = 10  # seconds between retries

llm_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)

# ---------- Rate limiter ----------
_llm_call_timestamps = deque()
MAX_CALLS_PER_MINUTE = 40
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_WAIT_TIMEOUT = 5
LAST_RATE_LIMIT = 0
RATE_LIMIT_COOLDOWN_SEC = 120

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

def deepseek_chat(prompt: str, system: str = "", retry: int = LLM_MAX_RETRIES) -> str:
    """Call LLM with retry on 429/503 errors."""
    global LAST_RATE_LIMIT
    for attempt in range(retry):
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
            print(f"LLM API error (attempt {attempt+1}/{retry}): {e}")
            if "429" in str(e) or "503" in str(e):
                LAST_RATE_LIMIT = time.time()
                delay = LLM_RETRY_DELAY * (attempt + 1)
                print(f"Retrying in {delay}s…")
                time.sleep(delay)
            else:
                break  # non‑retryable error
    return ""

# ---------- AI selects instruments ----------
def ai_pick_instruments() -> list:
    system = (
        "You are a senior financial market analyst. "
        "Select the 5 most important instruments to optimise trading strategies for today. "
        "Return ONLY a JSON array of OANDA instrument names, e.g. ['EUR_USD','XAU_USD']."
    )
    prompt = "What are the top 5 instruments to optimise trading strategies for today?"
    response = deepseek_chat(prompt, system)
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

# ---------- Symbol conversion ----------
def oanda_to_traderdev_symbol(oanda_symbol: str) -> str:
    return oanda_symbol.replace("_", "")

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

# ---------- Backtest workflow ----------
async def optimize_instrument(session, oanda_name: str, current_best: dict = None) -> dict:
    td_symbol = oanda_to_traderdev_symbol(oanda_name)
    current_sharpe = current_best.get("sharpe", -9999) if current_best else -9999

    print(f"\n📊 Optimizing {oanda_name} ({td_symbol})…")
    print(f"   Current best Sharpe: {current_sharpe:.3f}")

    system = (
        f"You are a Pine Script expert. Write a Pine Script v6 strategy specifically "
        f"for {td_symbol} on {TIMEFRAME} timeframe. "
        "ONLY use: ta.sma, ta.ema, ta.rsi, ta.macd, ta.crossover, ta.crossunder, "
        "ta.highest, ta.lowest, ta.atr, ta.bb. "
        "First line: //@version=6. "
        "Use strategy() with pyramiding=1, default_qty_type=strategy.percent_of_equity, "
        "default_qty_value=100. Include entry/exit with stop loss and take profit. "
        "Output ONLY code, no markdown."
    )
    user = f"Write a Pine Script v6 strategy for {td_symbol} {TIMEFRAME}."

    response = deepseek_chat(user, system)
    if not response:
        print("   ❌ LLM returned no strategy.")
        return None

    pine_code = clean_pine_code(response)
    name = f"github-{oanda_name}-{int(time.time())}"

    try:
        # Create strategy – the symbol is sometimes ignored, but we still pass it
        result = await session.call_tool(
            "create_strategy",
            arguments={"name": name, "symbol": td_symbol, "timeframe": TIMEFRAME, "pineSource": pine_code}
        )
        if not result.content:
            return None
        text = result.content[0].text
        if "error" in text.lower():
            print(f"   ⚠️ Create error: {text[:150]}")
            return None

        try:
            data = json.loads(text)
        except Exception:
            print(f"   ⚠️ Create response not JSON: {text[:200]}")
            return None
        sid = data.get("id")
        if not sid:
            return None

        # Run backtest – explicitly pass symbol to override default
        result = await session.call_tool(
            "run_backtest",
            arguments={
                "strategyId": sid,
                "symbol": td_symbol,           # explicitly set the symbol
                "timeframe": TIMEFRAME          # and the timeframe
            }
        )
        if not result.content:
            return None
        text = result.content[0].text
        if "error" in text.lower():
            print(f"   ⚠️ Backtest error: {text[:150]}")
            return None

        print(f"   Raw backtest response: {text[:500]}")
        sharpe = extract_sharpe(text)
        if sharpe is None:
            print("   ⚠️ Could not extract Sharpe.")
            return None

        print(f"   Sharpe = {sharpe:.3f}")

        if sharpe > current_sharpe:
            print(f"   ✅ Improved! {current_sharpe:.3f} → {sharpe:.3f}")
            return {
                "pine": pine_code,
                "sharpe": sharpe,
                "symbol": td_symbol,
                "optimized_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            }
        else:
            print(f"   No improvement (best: {current_sharpe:.3f})")
            return None

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

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
    print("🚀 Starting AI‑driven strategy optimization…")

    gist_id = GIST_ID
    if not gist_id:
        print("No GIST_ID set – creating a new gist…")
        gist_id = create_gist({
            "virtual_balance": 100.0,
            "trading_paused": False,
            "best_strategies": {}
        })
        print(f"✅ Created gist: {gist_id}")
        print(f"👉 Add this to your GitHub Actions secrets: GIST_ID = {gist_id}")

    try:
        state = read_gist(gist_id)
        best_strategies = state.get("best_strategies", {})
    except:
        best_strategies = {}

    instruments = ai_pick_instruments()
    if not instruments:
        instruments = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD", "US30_USD"]
        print("AI selection failed, using default list.")
    print(f"AI selected: {instruments}")

    headers = {}
    if TRADERDEV_API_KEY:
        headers["Authorization"] = f"Bearer {TRADERDEV_API_KEY}"

    async with sse_client(MCP_SERVER_URL, headers=headers) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            for oanda_name in instruments:
                current_best = best_strategies.get(oanda_name)
                result = await optimize_instrument(session, oanda_name, current_best)

                if result and "pine" in result:
                    best_strategies[oanda_name] = result
                    write_gist(gist_id, {
                        "virtual_balance": state.get("virtual_balance", 100.0),
                        "trading_paused": state.get("trading_paused", False),
                        "best_strategies": best_strategies,
                        "last_optimized": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                    })

                # Wait between instruments to stay under API limits
                await asyncio.sleep(5)

    print(f"\n🏁 Optimization finished. Strategies saved: {list(best_strategies.keys())}")

if __name__ == "__main__":
    asyncio.run(main())
