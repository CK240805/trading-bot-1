"""
Pre‑trading strategy optimization – DeepSeek + Trader.dev MCP
Generates Pine Script v6 strategies and backtests via quick_backtest.
Saves the best strategy and Sharpe ratio to a GitHub Gist.
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
def fix_pine_version(code: str) -> str:
    """Ensure the Pine Script starts with //@version=6."""
    # Replace any existing version directive
    code = re.sub(r'//@version\s*=\s*\d+', '//@version=6', code, count=1)
    # If no version directive at all, prepend it
    if not code.startswith('//@version'):
        code = '//@version=6\n' + code
    return code

# ---------- MCP helpers ----------
async def get_pine_rules(session) -> str:
    try:
        result = await session.call_tool("get_pine_codegen_rules", arguments={})
        if result.content and len(result.content) > 0:
            return result.content[0].text
    except Exception as e:
        print(f"Could not get Pine rules: {e}")
    return ""

async def backtest_pine_script(session, pine_code: str, instrument="EUR_USD", timeframe="H1") -> float:
    pine_code = fix_pine_version(pine_code)
    print(f"First line of Pine: {pine_code.split(chr(10))[0]}")
    try:
        result = await session.call_tool(
            "quick_backtest",
            arguments={
                "symbol": instrument,
                "timeframe": timeframe,
                "pineSource": pine_code
            }
        )
        if result.content and len(result.content) > 0:
            text = result.content[0].text
            print(f"Raw backtest response: {text[:500]}")
            # Check for errors
            if "error" in text.lower() or "mcprule_rejected" in text.lower():
                print(f"Backtest rejected: {text[:300]}")
                return 0.0
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

            # Get Pine Script rules
            pine_rules = await get_pine_rules(session)
            if pine_rules:
                print(f"Pine rules received ({len(pine_rules)} chars)")

            # Prompt for Pine Script v6
            system = (
                "You are a Pine Script expert. Generate a COMPLETE Pine Script v6 strategy "
                "that can be backtested. CRITICAL: the first line MUST be exactly:\n"
                "//@version=6\n"
                "Include entry/exit rules, stop loss, take profit. Use standard indicators. "
                "Output ONLY the Pine Script code, no markdown fences, no explanations.\n"
                f"Additional rules from the platform:\n{pine_rules}"
            )
            user = "Write a complete Pine Script v6 strategy for EUR/USD H1 with entry/exit rules and risk management."

            response = deepseek_chat(user, system)
            if not response:
                print("❌ LLM returned no strategy.")
                return

            # Clean up response
            if "```" in response:
                response = response.split("```")[1]
                if response.lower().startswith("pine"):
                    response = response[4:].strip()
            current_pine = response.strip()

            for i in range(2):
                print(f"Iteration {i+1}: Testing strategy…")
                sharpe = await backtest_pine_script(session, current_pine)
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

                if i < 1:  # Don't ask for improvement on the last iteration
                    prompt = (
                        f"The Pine Script below achieved a Sharpe of {sharpe:.3f}. "
                        f"Improve it to get a higher Sharpe. Return ONLY the improved Pine Script v6 code.\n\n"
                        f"Current code:\n{current_pine}"
                    )
                    response = deepseek_chat(prompt, system)
                    if not response:
                        break
                    if "```" in response:
                        response = response.split("```")[1]
                        if response.lower().startswith("pine"):
                            response = response[4:].strip()
                    current_pine = response.strip()

    print(f"🏁 Optimization finished. Best Sharpe: {best_sharpe:.3f}")

if __name__ == "__main__":
    asyncio.run(main())
