"""
Pre‑trading strategy optimization – DeepSeek + Trader.dev MCP
Creates strategy, runs backtest, polls for result, extracts Sharpe.
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
    """Remove markdown fences, ensure version 6, fix settings."""
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
    # Fix strategy settings
    code = re.sub(r'pyramiding\s*=\s*\d+', 'pyramiding=1', code)
    code = re.sub(r'default_qty_type\s*=\s*strategy\.\w+', 'default_qty_type=strategy.percent_of_equity', code)
    code = re.sub(r'default_qty_value\s*=\s*\d+', 'default_qty_value=100', code)
    code = re.sub(r'initial_capital\s*=\s*\d+', 'initial_capital=10000', code)
    return code.strip()

def generate_strategy_name() -> str:
    """Generate a unique strategy name."""
    return f"opt-{int(time.time())}"

# ---------- MCP workflow ----------
def extract_sharpe(text: str) -> float:
    """Deep search for Sharpe in backtest result."""
    try:
        data = json.loads(text)
        def find(obj, depth=0):
            if depth > 6:
                return None
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if "sharpe" in k.lower() and isinstance(v, (int, float)):
                        return float(v)
                    r = find(v, depth + 1)
                    if r is not None:
                        return r
            return None
        val = find(data)
        if val is not None:
            return val
    except:
        pass
    match = re.search(r'sharpe["\']?\s*[:=]\s*([0-9.]+)', text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 0.0

async def create_and_backtest(session, pine_code: str, symbol: str, timeframe: str) -> float:
    """Full workflow: create strategy → run backtest → poll result → return Sharpe."""
    pine_code = clean_pine_code(pine_code)
    strategy_name = generate_strategy_name()

    # Step 1: Create the strategy
    print(f"Creating strategy '{strategy_name}'…")
    try:
        result = await session.call_tool(
            "create_strategy",
            arguments={
                "name": strategy_name,
                "symbol": symbol,
                "timeframe": timeframe,
                "pineSource": pine_code
            }
        )
        if not result.content or len(result.content) == 0:
            print("Create strategy returned empty.")
            return 0.0

        text = result.content[0].text
        print(f"Create response: {text[:300]}")

        # Check for error
        if "error" in text.lower():
            print(f"Create strategy failed: {text[:200]}")
            return 0.0

        # Extract strategy ID
        strategy_id = None
        try:
            data = json.loads(text)
            strategy_id = data.get("id") or data.get("strategyId") or data.get("strategy_id")
        except:
            pass
        if not strategy_id:
            # Try to find any ID-like field
            match = re.search(r'"id"\s*:\s*"([^"]+)"', text)
            if match:
                strategy_id = match.group(1)
        if not strategy_id:
            print("Could not extract strategy ID.")
            return 0.0

        print(f"Strategy ID: {strategy_id}")
    except Exception as e:
        print(f"Create strategy error: {e}")
        return 0.0

    # Step 2: Run backtest
    print(f"Running backtest for {strategy_id}…")
    try:
        result = await session.call_tool(
            "run_backtest",
            arguments={"strategyId": strategy_id}
        )
        if not result.content or len(result.content) == 0:
            print("Run backtest returned empty.")
            return 0.0

        text = result.content[0].text
        print(f"Run backtest response: {text[:300]}")

        if "error" in text.lower():
            print(f"Run backtest failed: {text[:200]}")
            return 0.0

        # Extract job ID
        job_id = None
        try:
            data = json.loads(text)
            job_id = data.get("jobId") or data.get("id") or data.get("job_id")
        except:
            pass
        if not job_id:
            match = re.search(r'"jobId"\s*:\s*"([^"]+)"', text)
            if match:
                job_id = match.group(1)
        if not job_id:
            print("Could not extract job ID, trying to get result directly…")
            # Some APIs return the result immediately
            return extract_sharpe(text)

        print(f"Job ID: {job_id}")
    except Exception as e:
        print(f"Run backtest error: {e}")
        return 0.0

    # Step 3: Poll for result
    print(f"Polling for result of job {job_id}…")
    for attempt in range(15):
        await asyncio.sleep(2)
        try:
            result = await session.call_tool(
                "get_backtest_result",
                arguments={"jobId": job_id}
            )
            if result.content and len(result.content) > 0:
                text = result.content[0].text
                if "USER HINT" in text or "pending" in text.lower() or "running" in text.lower():
                    print(f"  Attempt {attempt+1}: still running…")
                    continue
                if "error" in text.lower():
                    print(f"Backtest error: {text[:300]}")
                    return 0.0
                print(f"Result: {text[:400]}")
                return extract_sharpe(text)
        except Exception as e:
            print(f"Poll error: {e}")
    print("Backtest timed out.")
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

            system = (
                "You are a Pine Script expert. Write a COMPLETE Pine Script v6 strategy. "
                "RULES:\n"
                "1. First line MUST be: //@version=6\n"
                "2. Use strategy() with pyramiding=1, default_qty_type=strategy.percent_of_equity, default_qty_value=100\n"
                "3. Include strategy.entry() and strategy.exit() with stop loss and take profit.\n"
                "4. Use standard indicators: ta.sma, ta.rsi, ta.macd, ta.ema.\n"
                "5. Output ONLY the code, no markdown, no explanations."
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
                        f"Pine Script achieved Sharpe={sharpe:.3f}. Improve it. Return ONLY improved v6 code.\n\n"
                        f"Current:\n{current_pine}"
                    )
                    response = deepseek_chat(prompt, system)
                    if not response:
                        break
                    current_pine = clean_pine_code(response)

    print(f"\n🏁 Optimization finished. Best Sharpe: {best_sharpe:.3f}")

if __name__ == "__main__":
    asyncio.run(main())
