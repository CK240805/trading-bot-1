"""
Pre‑trading strategy optimization – DeepSeek + Trader.dev MCP
Dynamically cross‑matches OANDA crypto pairs with Trader.dev symbols
by querying search_perps with each base currency.
Saves per‑instrument best strategies to a GitHub Gist.
"""
import os, json, time, asyncio, requests, re
from collections import deque
from openai import OpenAI
from mcp import ClientSession
from mcp.client.sse import sse_client

# OANDA (to fetch instrument list)
from oandapyV20 import API
import oandapyV20.endpoints.accounts as accounts

# ---------- Config ----------
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-ai/deepseek-v4-flash")
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "https://mcp.trader.dev/sse")
TRADERDEV_API_KEY = os.environ.get("TRADERDEV_API_KEY", "")
GITHUB_GIST_TOKEN = os.environ["GITHUB_GIST_TOKEN"]
GIST_ID = os.environ.get("GIST_ID")
OANDA_ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID", "")
OANDA_API_KEY = os.environ.get("OANDA_API_KEY", "")
OANDA_ENV = os.environ.get("OANDA_ENV", "practice")

TIMEFRAME = "1h"
LLM_MAX_RETRIES = 5
LLM_RETRY_DELAY = 10
MCP_MAX_RETRIES = 3
MCP_RETRY_DELAY = 5

llm_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
oanda_api = API(access_token=OANDA_API_KEY, environment=OANDA_ENV) if OANDA_API_KEY else None

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

# ---------- MCP tool call with retry ----------
async def mcp_call_tool_with_retry(session, tool_name: str, arguments: dict, max_retries: int = MCP_MAX_RETRIES, delay: int = MCP_RETRY_DELAY) -> dict:
    last_exception = None
    for attempt in range(max_retries):
        try:
            result = await session.call_tool(tool_name, arguments=arguments)
            return result
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait = delay * (attempt + 1)
                print(f"   MCP tool '{tool_name}' failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s…")
                await asyncio.sleep(wait)
    raise last_exception

# ---------- Fetch instruments ----------
def get_oanda_instruments() -> set:
    """Return a set of OANDA instrument names."""
    if not oanda_api or not OANDA_ACCOUNT_ID:
        print("⚠️ OANDA credentials not set – using fallback list.")
        return {"BTC_USD", "ETH_USD", "LTC_USD", "BCH_USD", "XRP_USD"}
    try:
        r = accounts.AccountInstruments(accountID=OANDA_ACCOUNT_ID)
        resp = oanda_api.request(r)
        return {instr["name"] for instr in resp.get("instruments", [])}
    except Exception as e:
        print(f"Failed to fetch OANDA instruments: {e}")
        return {"BTC_USD", "ETH_USD", "LTC_USD", "BCH_USD", "XRP_USD"}

async def debug_search_perps(session):
    """Test search_perps with 'BTC' and print raw response."""
    try:
        result = await mcp_call_tool_with_retry(session, "search_perps", {"query": "BTC"})
        if result.content:
            print("🔍 Raw search_perps response for 'BTC' (first 2000 chars):")
            print(result.content[0].text[:2000])
    except Exception as e:
        print(f"Debug search failed: {e}")

async def match_oanda_crypto_to_traderdev(session, oanda_set: set) -> dict:
    """
    For every OANDA crypto pair (ending with _USD), extract the base currency
    and search Trader.dev for matching symbols. Returns a dict mapping
    OANDA name → Trader.dev symbol (e.g. 'BTC_USD' → 'BTCUSDT').
    """
    matched = {}
    oanda_crypto = [name for name in oanda_set if name.endswith("_USD")]

    for oanda_name in oanda_crypto:
        base = oanda_name.replace("_USD", "")  # e.g. "BTC"
        print(f"   Searching Trader.dev for base '{base}'…")
        try:
            result = await mcp_call_tool_with_retry(
                session, "search_perps", {"query": base}
            )
            if result.content and result.content[0].text:
                text = result.content[0].text
                data = json.loads(text)
                symbols = []
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "symbol" in item:
                            symbols.append(item["symbol"])
                        elif isinstance(item, str):
                            symbols.append(item)
                # Try to find a symbol that contains the base (case‑insensitive) and ends with USDT
                for sym in symbols:
                    if base.upper() in sym.upper() and sym.upper().endswith("USDT"):
                        matched[oanda_name] = sym
                        print(f"   ✅ Matched {oanda_name} → {sym}")
                        break
        except Exception as e:
            print(f"   ⚠️ Could not search for {base}: {e}")

    return matched

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

def extract_sharpe(obj):
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except:
            matches = re.findall(r'"sharpe[^"]*"\s*:\s*([0-9.-]+)', obj, re.IGNORECASE)
            if matches:
                return float(matches[0])
            return None
    def _search(o, depth=0):
        if depth > 10 or o is None:
            return None
        if isinstance(o, dict):
            for k, v in o.items():
                if "sharpe" in k.lower() and isinstance(v, (int, float)):
                    return float(v)
                r = _search(v, depth + 1)
                if r is not None:
                    return r
        elif isinstance(o, list):
            for item in o:
                r = _search(item, depth + 1)
                if r is not None:
                    return r
        return None
    return _search(obj)

# ---------- Backtest workflow ----------
async def optimize_instrument(session, oanda_name: str, td_symbol: str, current_best: dict = None) -> dict:
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
        result = await mcp_call_tool_with_retry(
            session, "create_strategy",
            {"name": name, "symbol": td_symbol, "timeframe": TIMEFRAME, "pineSource": pine_code}
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

        result = await mcp_call_tool_with_retry(
            session, "run_backtest",
            {"strategyId": sid, "symbol": td_symbol, "timeframe": TIMEFRAME}
        )
        if not result.content:
            return None
        text = result.content[0].text
        if "error" in text.lower():
            print(f"   ⚠️ Backtest error: {text[:150]}")
            return None

        print(f"   Raw backtest response (full):\n{text}\n")

        sharpe = extract_sharpe(text)
        if sharpe is None:
            print("   ⚠️ Could not extract Sharpe (raw response printed above).")
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
    print("🚀 Starting instrument‑matched strategy optimization…")

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

    headers = {}
    if TRADERDEV_API_KEY:
        headers["Authorization"] = f"Bearer {TRADERDEV_API_KEY}"

    async with sse_client(MCP_SERVER_URL, headers=headers) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Debug: see what search_perps returns for BTC
            await debug_search_perps(session)

            # 1. Fetch OANDA instruments
            print("📡 Fetching OANDA instruments…")
            oanda_set = get_oanda_instruments()
            print(f"   Found {len(oanda_set)} instruments on OANDA.")

            # 2. Dynamically match crypto pairs with Trader.dev
            print("🔎 Matching OANDA crypto pairs to Trader.dev…")
            matched = await match_oanda_crypto_to_traderdev(session, oanda_set)

            if not matched:
                print("⚠️ No matches found, using hardcoded crypto fallback.")
                matched = {
                    "BTC_USD": "BTCUSDT",
                    "ETH_USD": "ETHUSDT",
                    "LTC_USD": "LTCUSDT",
                    "BCH_USD": "BCHUSDT",
                    "XRP_USD": "XRPUSDT"
                }

            print(f"🔗 Matched {len(matched)} instruments:")
            for k, v in matched.items():
                print(f"   {k} → {v}")

            # 3. Optimize each matched instrument
            for oanda_name, td_symbol in matched.items():
                current_best = best_strategies.get(oanda_name)
                result = await optimize_instrument(session, oanda_name, td_symbol, current_best)

                if result and "pine" in result:
                    best_strategies[oanda_name] = result
                    write_gist(gist_id, {
                        "virtual_balance": state.get("virtual_balance", 100.0),
                        "trading_paused": state.get("trading_paused", False),
                        "best_strategies": best_strategies,
                        "last_optimized": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                    })

                await asyncio.sleep(5)

    print(f"\n🏁 Optimization finished. Strategies saved: {list(best_strategies.keys())}")

if __name__ == "__main__":
    asyncio.run(main())
