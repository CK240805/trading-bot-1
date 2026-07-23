"""
Pre‑trading strategy optimization – DeepSeek + Trader.dev MCP
Creates strategy, runs backtest, extracts Sharpe from inline result.
Saves the best strategy to a GitHub Gist.
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

INSTRUMENT = "EURUSD"
TIMEFRAME = "1h"

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

def deepseek_chat(prompt: str, system: str = "") -> str:
    global LAST_RATE_LIMIT
    now = time.time()
    if now - LAST_RATE_LIMIT < RATE_LIMIT_COOLDOWN_SEC:
        print("LLM call skipped – cooldown active.")
        return ""
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
        print(f"LLM API error: {e}")
        if "429" in str(e) or "503" in str(e):
            LAST_RATE_LIMIT = time.time()
        return ""

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
    code = re.sub(r'initial_capital\s*=\s*\d+', 'initial_capital=10000', code)
    return code.strip()

def generate_strategy_name() -> str:
    return f"opt-{int(time.time())}"

# ---------- Sharpe extraction ----------
def extract_sharpe(obj, depth=0):
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
async def create_and_backtest(session, pine_code: str, symbol: str, timeframe: str) -> float:
    pine_code = clean_pine_code(pine_code)
    strategy_name = generate_strategy_name()

    result = await session.call_tool(
        "create_strategy",
        arguments={"name": strategy_name, "symbol": symbol, "timeframe": timeframe, "pineSource": pine_code}
    )
    if not result.content:
        return 0.0
    text = result.content[0].text
    if "error" in text.lower():
        return 0.0

    data = json.loads(text)
    strategy_id = data.get("id")

    result = await session.call_tool("run_backtest", arguments={"strategyId": strategy_id})
    if not result.content:
        return 0.0
    text = result.content[0].text
    if "error" in text.lower():
        # Print the error so we can add more unsupported functions to the prompt
        try:
            err = json.loads(text)
            print(f"  Backtest error: {err.get('message', text[:200])}")
        except:
            print(f"  Backtest error: {text[:200]}")
        return 0.0

    data = json.loads(text)
    sharpe = extract_sharpe(data)
    if sharpe is not None:
        return sharpe
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
    print("🚀 Starting strategy optimization…")

    gist_id = GIST_ID
    if not gist_id:
        print("No GIST_ID set – creating a new gist…")
        gist_id = create_gist({"best_strategy": None, "best_sharpe": -9999, "last_optimized": None})
        print(f"✅ Created gist: {gist_id}")
        print(f"👉 Add this to your GitHub Actions secrets: GIST_ID = {gist_id}")

    try:
        state = read_gist(gist_id)
        best_strategy = state.get("best_strategy")
        best_sharpe = state.get("best_sharpe", -9999)
        print(f"Previous best: Sharpe={best_sharpe:.3f}")
    except Exception as e:
        print(f"Could not read gist: {e} – starting fresh.")
        best_strategy = None
        best_sharpe = -9999

    headers = {}
    if TRADERDEV_API_KEY:
        headers["Authorization"] = f"Bearer {TRADERDEV_API_KEY}"

    async with sse_client(MCP_SERVER_URL, headers=headers) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Updated prompt with list of KNOWN SUPPORTED functions
            system = (
                "You are a Pine Script expert. Write a COMPLETE and VALID Pine Script v6 strategy.\n"
                "CRITICAL RULES:\n"
                "1. First line: //@version=6\n"
                "2. Use strategy() with pyramiding=1, default_qty_type=strategy.percent_of_equity, default_qty_value=100\n"
                "3. Include strategy.entry() and strategy.exit() with stop loss and take profit.\n"
                "4. ONLY use these supported functions (others will crash):\n"
                "   ta.sma, ta.ema, ta.rsi, ta.macd, ta.crossover, ta.crossunder,\n"
                "   ta.highest, ta.lowest, ta.atr, ta.stdev, ta.bb, ta.supertrend,\n"
                "   request.security, math.abs, math.max, math.min\n"
                "5. DO NOT use: ta.adx, ta.cci, ta.mfi, ta.obv, ta.stoch, ta.psar, ta.vwap,\n"
                "   ta.wma, ta.vwma, ta.rma, ta.alma (these are NOT supported)\n"
                "6. Output ONLY the code, no markdown, no explanations."
            )
            user = f"Write a Pine Script v6 strategy for {INSTRUMENT} {TIMEFRAME}."

            response = deepseek_chat(user, system)
            if not response:
                print("❌ LLM returned no strategy.")
                return

            current_pine = clean_pine_code(response)

            for i in range(2):
                print(f"\nIteration {i+1}: Testing strategy…")
                sharpe = await create_and_backtest(session, current_pine, INSTRUMENT, TIMEFRAME)
                print(f"  Sharpe = {sharpe:.3f}")

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_strategy = current_pine
                    write_gist(gist_id, {
                        "best_strategy": best_strategy,
                        "best_sharpe": best_sharpe,
                        "last_optimized": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                    })
                    print(f"  ✅ New best saved! Sharpe = {best_sharpe:.3f}")
                    if best_sharpe >= 2.0:
                        print("🎉 Amazing strategy found!")
                        break

                if i < 1:
                    prompt = (
                        f"Pine Script achieved Sharpe={sharpe:.3f}. Improve it using only supported functions. "
                        f"Return ONLY improved v6 code.\n\nCurrent:\n{current_pine}"
                    )
                    response = deepseek_chat(prompt, system)
                    if not response:
                        break
                    current_pine = clean_pine_code(response)

    print(f"\n🏁 Optimization finished. Best Sharpe: {best_sharpe:.3f}")

if __name__ == "__main__":
    asyncio.run(main())
