"""
Pre‑trading strategy optimization – DeepSeek + Trader.dev MCP
Saves the best strategy and Sharpe ratio to a GitHub Gist.
Run by GitHub Actions before trading starts.
Includes rate limiter (max 40 calls/min) and 503 cooldown.
"""
import os, json, time, asyncio, requests, traceback
from collections import deque
from openai import OpenAI
from mcp import ClientSession
from mcp.client.sse import sse_client

# ---------- Config from environment ----------
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-ai/deepseek-v4-flash")
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "https://mcp.trader.dev/sse")
TRADERDEV_API_KEY = os.environ.get("TRADERDEV_API_KEY", "")
GITHUB_GIST_TOKEN = os.environ["GITHUB_GIST_TOKEN"]
GIST_ID = os.environ.get("GIST_ID")

# NVIDIA client
llm_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)

# ---------- Rate limiter (same as trading bot) ----------
_llm_call_timestamps = deque()
MAX_CALLS_PER_MINUTE = 40
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_WAIT_TIMEOUT = 5
LAST_RATE_LIMIT = 0
RATE_LIMIT_COOLDOWN_SEC = 120

def _check_rate_limit() -> bool:
    """Enforce max 40 LLM calls per minute. Wait up to 5 seconds if at limit."""
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
        print("Rate limit reached, skipping LLM call (wait too long).")
        return False
    print(f"Rate limit reached, waiting {wait_time:.1f}s…")
    time.sleep(wait_time)
    _llm_call_timestamps.popleft()
    _llm_call_timestamps.append(time.time())
    return True

# ---------- LLM call ----------
def deepseek_chat(prompt: str, system: str = "") -> str:
    global LAST_RATE_LIMIT
    now = time.time()
    if now - LAST_RATE_LIMIT < RATE_LIMIT_COOLDOWN_SEC:
        print("LLM call skipped – rate limit cooldown active (503).")
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

# ---------- MCP backtest ----------
async def backtest_strategy(strategy: dict, instrument="EUR_USD", timeframe="H1") -> float:
    headers = {}
    if TRADERDEV_API_KEY:
        headers["Authorization"] = f"Bearer {TRADERDEV_API_KEY}"

    try:
        async with sse_client(MCP_SERVER_URL, headers=headers) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = await session.list_tools()
                tool_names = []
                if hasattr(tools, 'tools'):
                    tool_names = [t.name for t in tools.tools]
                elif isinstance(tools, list):
                    for t in tools:
                        if hasattr(t, 'name'):
                            tool_names.append(t.name)
                print(f"Available tools: {tool_names}")

                backtest_tool = None
                if "quick_backtest" in tool_names:
                    backtest_tool = "quick_backtest"
                elif "run_backtest" in tool_names:
                    backtest_tool = "run_backtest"
                else:
                    print("❌ No backtest tool found!")
                    return 0.0

                print(f"Using tool: {backtest_tool}")
                args = {"symbol": instrument, "timeframe": timeframe, "strategy": strategy}
                result = await session.call_tool(backtest_tool, arguments=args)

                if result.content and len(result.content) > 0:
                    text = result.content[0].text
                    print(f"Raw backtest response: {text[:500]}")
                    try:
                        data = json.loads(text)
                        sharpe = data.get("sharpe") or data.get("sharpe_ratio") or \
                                 data.get("performance", {}).get("sharpe")
                        if sharpe is not None:
                            return float(sharpe)
                    except:
                        pass
                    try:
                        return float(text.strip())
                    except:
                        pass
                return 0.0
    except Exception as e:
        print(f"Backtest error: {e}")
        if hasattr(e, 'exceptions'):
            for sub_exc in e.exceptions:
                print(f"  Inner exception: {sub_exc}")
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
    resp = requests.patch(f"https://api.github.com/gists/{gist_id}", headers=GIST_HEADERS, json=payload)
    resp.raise_for_status()

def create_gist(data: dict) -> str:
    payload = {
        "description": "Trading bot state",
        "public": False,
        "files": {"bot_state.json": {"content": json.dumps(data, indent=2)}}
    }
    resp = requests.post("https://api.github.com/gists", headers=GIST_HEADERS, json=payload)
    resp.raise_for_status()
    return resp.json()["id"]

# ---------- Main optimization loop ----------
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

    system = (
        "You are a quant strategist. Output a JSON strategy object that Trader.dev can backtest. "
        "Example: {\"type\": \"sma_crossover\", \"fast_ma\": 10, \"slow_ma\": 30, \"stop_loss_pct\": 2}. "
        "Only JSON."
    )
    user = "Propose an initial trading strategy for EUR/USD H1."
    response = deepseek_chat(user, system)
    if not response:
        print("❌ LLM returned no strategy.")
        return
    try:
        current = json.loads(response)
    except:
        print("❌ Could not parse strategy JSON.")
        return

    for i in range(3):   # Reduced from 5 to 3 to save API calls
        print(f"Iteration {i+1}: Testing strategy…")
        sharpe = await backtest_strategy(current)
        print(f"  Sharpe = {sharpe:.3f}")

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_strategy = current
            write_gist(gist_id, {
                "best_strategy": best_strategy,
                "best_sharpe": best_sharpe,
                "last_optimized": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            })
            print(f"  ✅ New best saved! Sharpe = {best_sharpe:.3f}")
            if best_sharpe >= 2.0:
                print("🎉 Amazing strategy found!")
                break

        prompt = f"The last strategy (Sharpe {sharpe:.3f}) was: {json.dumps(current)}. Propose an improved version. Only JSON."
        response = deepseek_chat(prompt, system)
        if not response:
            break
        try:
            current = json.loads(response)
        except:
            break

    print(f"🏁 Optimization finished. Best Sharpe: {best_sharpe:.3f}")

if __name__ == "__main__":
    asyncio.run(main())
