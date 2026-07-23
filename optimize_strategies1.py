"""
OANDA Strategy Optimizer – DeepSeek decides everything.
DeepSeek selects:
  - Which 5 instruments to optimise today
  - For each instrument: the strategy type (sma_cross / rsi_reversal / macd_cross)
  - The strategy parameters as JSON
Backtests via local OANDA MCP server and saves best per‑instrument strategy to Gist.
"""
import os, json, time, asyncio, requests, re
from collections import deque
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---------- Config ----------
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-ai/deepseek-v4-flash")
GITHUB_GIST_TOKEN = os.environ["GITHUB_GIST_TOKEN"]
GIST_ID = os.environ.get("GIST_ID")

MAX_INSTRUMENTS_PER_RUN = int(os.environ.get("MAX_INSTRUMENTS_PER_RUN", "5"))
LLM_MAX_RETRIES = 5
LLM_RETRY_DELAY = 10

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
    """Ask DeepSeek which 5 OANDA instruments to optimise today."""
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

def ai_propose_strategy(instrument: str) -> dict:
    """
    Ask DeepSeek to propose a strategy type and parameters for the given instrument.
    Returns a dict with keys: 'type' (strategy_type), 'params' (parameter dict).
    """
    system = (
        "You are an expert quantitative strategist. For the given OANDA instrument, "
        "choose the best strategy type among: sma_cross, rsi_reversal, macd_cross. "
        "Then provide appropriate parameters. "
        "Return ONLY a JSON object with exactly two keys: "
        '"type": one of the three strategy types, '
        '"params": a JSON object with the required parameters for that strategy.\n\n'
        "Examples:\n"
        '  sma_cross: {"type":"sma_cross","params":{"ma_fast":10,"ma_slow":30}}\n'
        '  rsi_reversal: {"type":"rsi_reversal","params":{"rsi_period":14,"oversold":30,"overbought":70}}\n'
        '  macd_cross: {"type":"macd_cross","params":{"fast":12,"slow":26,"signal":9}}'
    )
    prompt = f"Instrument: {instrument}\nTimeframe: H1"
    response = deepseek_chat(prompt, system)
    if not response:
        return None
    try:
        if "```" in response:
            response = response.split("```")[1].replace("json", "").strip()
        proposal = json.loads(response)
        if "type" in proposal and "params" in proposal:
            return proposal
    except:
        pass
    return None

# ---------- MCP client helpers ----------
SERVER_PARAMS = StdioServerParameters(command="python", args=["oanda_mcp_server.py"])

async def backtest(session, instrument: str, strategy_type: str, params: dict) -> float:
    result = await session.call_tool("backtest_strategy", {
        "instrument": instrument,
        "strategy_type": strategy_type,
        "params": params
    })
    if result.content:
        data = json.loads(result.content[0].text)
        if "error" in data:
            print(f"   Backtest error: {data['error']}")
            return 0.0
        return data.get("sharpe", 0.0)
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
    print("🚀 Starting AI‑driven OANDA strategy optimization…")

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

    # AI picks instruments
    instruments = ai_pick_instruments()
    if not instruments:
        instruments = ["EUR_USD", "GBP_USD", "USD_JPY", "XAU_USD", "US30_USD"]
        print("AI selection failed, using default list.")
    print(f"AI selected instruments: {instruments}")

    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            for instrument in instruments[:MAX_INSTRUMENTS_PER_RUN]:
                current_best = best_strategies.get(instrument, {})
                current_sharpe = current_best.get("sharpe", -9999)
                print(f"\n📊 Optimizing {instrument} (best Sharpe: {current_sharpe:.3f})…")

                proposal = ai_propose_strategy(instrument)
                if not proposal:
                    print("   ❌ Could not get strategy proposal.")
                    continue

                strategy_type = proposal["type"]
                params = proposal["params"]
                print(f"   Strategy: {strategy_type} with params {params}")

                sharpe = await backtest(session, instrument, strategy_type, params)
                print(f"   Sharpe = {sharpe:.3f}")

                if sharpe > current_sharpe:
                    best_strategies[instrument] = {
                        "type": strategy_type,
                        "params": params,
                        "sharpe": sharpe,
                        "optimized_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                    }
                    print(f"   ✅ Improved! New best Sharpe: {sharpe:.3f}")

                    # Save to Gist
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
