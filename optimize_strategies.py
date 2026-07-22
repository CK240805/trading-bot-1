"""
Pre‑trading strategy optimization – DeepSeek + Trader.dev MCP
Saves the best strategy and Sharpe ratio to a GitHub Gist.
Run by GitHub Actions before trading starts.
"""
import os, json, time, asyncio, requests, traceback
from openai import OpenAI
from mcp import ClientSession
from mcp.client.sse import sse_client

# ---------- Config from environment ----------
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-ai/deepseek-v4-flash")
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "https://mcp.trader.dev/sse")
TRADERDEV_API_KEY = os.environ.get("TRADERDEV_API_KEY", "")
GITHUB_GIST_TOKEN = os.environ["GITHUB_GIST_TOKEN"]
GIST_ID = os.environ.get("GIST_ID")  # optional – if not set, a new gist will be created

# NVIDIA client
llm_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)

# ---------- LLM call ----------
def deepseek_chat(prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = llm_client.chat.completions.create(
        model=LLM_MODEL, messages=messages,
        temperature=1, top_p=0.95, max_tokens=16384, stream=False
    )
    return resp.choices[0].message.content

# ---------- MCP backtest (with detailed error logging) ----------
async def backtest_strategy(strategy: dict, instrument="EUR_USD", timeframe="H1") -> float:
    headers = {}
    if TRADERDEV_API_KEY:
        headers["Authorization"] = f"Bearer {TRADERDEV_API_KEY}"

    try:
        async with sse_client(MCP_SERVER_URL, headers=headers) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # First, list available tools to see what we have
                tools = await session.list_tools()
                tool_names = []
                if hasattr(tools, 'tools'):
                    tool_names = [t.name for t in tools.tools]
                elif isinstance(tools, list):
                    for t in tools:
                        if hasattr(t, 'name'):
                            tool_names.append(t.name)
                print(f"Available tools: {tool_names}")

                # Try quick_backtest first, fall back to run_backtest
                backtest_tool = None
                if "quick_backtest" in tool_names:
                    backtest_tool = "quick_backtest"
                elif "run_backtest" in tool_names:
                    backtest_tool = "run_backtest"
                else:
                    print("❌ No backtest tool found!")
                    return 0.0

                print(f"Using tool: {backtest_tool}")

                # Build arguments – try different formats
                args = {"symbol": instrument, "timeframe": timeframe}
                # Try passing strategy as JSON string or object
                args["strategy"] = strategy  # some servers want the raw object

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
                    # Try parsing as plain number
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
                traceback.print_exception(type(sub_exc), sub_exc, sub_exc.__traceback__)
        else:
            traceback.print_exc()
        return 0.0

# ---------- Gist helpers ----------
GIST_HEADERS = {
    "Authorization": f"token {GITHUB_GIST_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def read_gist(gist_id: str) -> dict:
    """Read bot_state.json from a gist."""
    resp = requests.get(f"https://api.github.com/gists/{gist_id}", headers=GIST_HEADERS)
    resp.raise_for_status()
    gist = resp.json()
    content = gist["files"].get("bot_state.json", {}).get("content", "{}")
    return json.loads(content)

def write_gist(gist_id: str, data: dict):
    """Write bot_state.json to a gist."""
    payload = {"files": {"bot_state.json": {"content": json.dumps(data, indent=2)}}}
    resp = requests.patch(f"https://api.github.com/gists/{gist_id}", headers=GIST_HEADERS, json=payload)
    resp.raise_for_status()

def create_gist(data: dict) -> str:
    """Create a new gist and return its ID."""
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

    for i in range(5):
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
