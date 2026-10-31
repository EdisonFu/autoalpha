import os
import json
from typing import List, Dict, Any


# 兼容既有部署的密钥变量名 → (协议, 默认 base_url)
# 具体模型 ID 一律由 LLM_MODEL 指定，此处不写死任何型号。
_KEY_ENVS = [
    ("LLM_API_KEY",       "anthropic", ""),
    ("ANTHROPIC_API_KEY", "anthropic", ""),
    ("DEEPSEEK_API_KEY",  "openai",    "https://api.deepseek.com/v1"),
    ("OPENAI_API_KEY",    "openai",    ""),
]


class CreditExhaustedError(RuntimeError):
    """Raised when the API account has insufficient credits."""


# ── Token 计量 ───────────────────────────────────────────────────────────────
# 价格（美元 / 百万 input / output token），用于把 token 数换算成花费。
#
# 不硬编码任何模型名：价格通过环境变量提供，未配置时按 0 计价并在汇总里
# 标记 priced=False（用量仍然准确，只是不折算金额）。
#
#   export LLM_PRICE_INPUT=3.0     # 美元 / 百万 input token
#   export LLM_PRICE_OUTPUT=15.0   # 美元 / 百万 output token
#
# 也可用 LLM_PRICING 传 JSON 做多模型映射：
#   export LLM_PRICING='{"<model-id>": [3.0, 15.0]}'

def _load_pricing() -> dict:
    raw = os.environ.get("LLM_PRICING", "").strip()
    if raw:
        try:
            return {k: tuple(v) for k, v in json.loads(raw).items()}
        except Exception:
            pass
    return {}


_PRICING = _load_pricing()


def _price_for(model: str):
    """返回 (input_price, output_price)，单位 美元/百万token。未知则 (0,0)。"""
    if model in _PRICING:
        return _PRICING[model]
    try:
        pin = float(os.environ.get("LLM_PRICE_INPUT", "") or 0)
        pout = float(os.environ.get("LLM_PRICE_OUTPUT", "") or 0)
    except ValueError:
        return (0.0, 0.0)
    return (pin, pout)


class TokenMeter:
    """累计 token 用量，按调用目的分类。线程安全（沙箱/并发下会被多线程写）。"""

    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.by_purpose: Dict[str, Dict[str, int]] = {}

    def record(self, purpose: str, inp: int, out: int, model: str = "") -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += inp
            self.output_tokens += out
            d = self.by_purpose.setdefault(
                purpose, {"calls": 0, "input": 0, "output": 0})
            d["calls"] += 1
            d["input"] += inp
            d["output"] += out
            self._model = model or getattr(self, "_model", "")

    def cost_usd(self, model: str = "") -> float:
        m = model or getattr(self, "_model", "")
        pin, pout = _price_for(m)
        return (self.input_tokens * pin + self.output_tokens * pout) / 1e6

    def summary(self, model: str = "") -> dict:
        m = model or getattr(self, "_model", "")
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "cost_usd": round(self.cost_usd(m), 4),
            "model": m,
            "priced": any(_price_for(m)),
            "by_purpose": dict(self.by_purpose),
        }


class LLMClient:
    def __init__(self, config: dict = None):
        """
        config 可选。不传时从环境变量自动初始化，见 _auto_config()。
        模型 ID 由 LLM_MODEL 指定，代码中不写死任何型号。
        """
        if config is None:
            config = self._auto_config()
        self.config = config
        self.model = config["model"]
        provider = config.get("provider", "anthropic")
        api_key = os.environ.get(config.get("api_key_env", "LLM_API_KEY"), "")

        self.meter = TokenMeter()

        if provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.provider = "anthropic"
        else:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url=config.get("base_url", "https://api.openai.com/v1"),
            )
            self.provider = "openai"

    @staticmethod
    def _auto_config() -> dict:
        """
        从环境变量推断配置。模型 ID 一律由 LLM_MODEL 提供，不在代码中写死，
        这样换模型/换供应商不需要改代码，仓库里也不留具体型号。

          LLM_API_KEY   密钥（也兼容各家默认的 *_API_KEY 变量名）
          LLM_BASE_URL  OpenAI 兼容端点；设置即走 openai 协议
          LLM_MODEL     模型 ID
        """
        model = os.environ.get("LLM_MODEL", "").strip()
        base_url = os.environ.get("LLM_BASE_URL", "").strip()

        if os.environ.get("LLM_API_KEY"):
            cfg = {"provider": "openai" if base_url else "anthropic",
                   "model": model, "api_key_env": "LLM_API_KEY"}
            if base_url:
                cfg["base_url"] = base_url
            if not model:
                raise EnvironmentError("设置 LLM_API_KEY 时必须同时设置 LLM_MODEL")
            return cfg

        # 兼容既有的各家默认变量名
        for env, provider, url in _KEY_ENVS:
            if os.environ.get(env):
                cfg = {"provider": provider,
                       "model": model or os.environ.get(f"{env}_MODEL", ""),
                       "api_key_env": env}
                if url:
                    cfg["base_url"] = url
                if not cfg["model"]:
                    raise EnvironmentError(
                        f"检测到 {env}，但未指定模型。请设置 LLM_MODEL=<model-id>")
                return cfg

        raise EnvironmentError(
            "未找到 API 密钥。请设置 LLM_API_KEY 与 LLM_MODEL"
            "（如使用 OpenAI 兼容端点，再设置 LLM_BASE_URL）"
        )

    def complete(self, system: str = "", user: str = "", prompt: str = "",
                 temperature: float = None, purpose: str = "other") -> str:
        """
        Single-turn completion. 支持两种调用方式：
          complete(system="...", user="...")              # 推荐
          complete(prompt="...", system="...")            # 兼容旧调用
          complete(..., temperature=0.3)                 # 修复模式低温度（来自 BrainAlpha/AlphaEvolve）
        temperature=None 时使用模型默认值；生成用 ~0.7，修复用 ~0.3。
        """
        content = user or prompt
        if self.provider == "anthropic":
            kwargs: Dict[str, Any] = dict(
                model=self.model,
                max_tokens=8192,
                system=system or "You are a helpful assistant.",
                messages=[{"role": "user", "content": content}],
            )
            if temperature is not None:
                kwargs["temperature"] = temperature
            try:
                msg = self.client.messages.create(**kwargs)
            except Exception as e:
                es = str(e).lower()
                if "credit balance" in es or "insufficient" in es:
                    raise CreditExhaustedError(str(e)) from e
                raise  # connection errors etc. propagate as-is
            u = getattr(msg, "usage", None)
            if u is not None:
                self.meter.record(purpose, getattr(u, "input_tokens", 0) or 0,
                                  getattr(u, "output_tokens", 0) or 0, self.model)
            return msg.content[0].text
        else:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": content})
            kwargs = dict(model=self.model, messages=messages, max_tokens=8192)
            if temperature is not None:
                kwargs["temperature"] = temperature
            resp = self.client.chat.completions.create(**kwargs)
            u = getattr(resp, "usage", None)
            if u is not None:
                self.meter.record(purpose, getattr(u, "prompt_tokens", 0) or 0,
                                  getattr(u, "completion_tokens", 0) or 0, self.model)
            return resp.choices[0].message.content

    def chat_with_tools(self, messages: List[Dict], system: str, tools: List[Dict]) -> Dict:
        """
        Multi-turn chat with tool calling.
        Returns:
          {"type": "text", "content": str, "raw": ...}
          {"type": "tool_use", "tool_name": str, "tool_input": dict, "tool_use_id": str, "raw": ...}
        """
        if self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                system=system,
                messages=messages,
                tools=tools,
            )
            for block in resp.content:
                if block.type == "tool_use":
                    return {
                        "type": "tool_use",
                        "tool_name": block.name,
                        "tool_input": block.input,
                        "tool_use_id": block.id,
                        "raw": resp,
                    }
            text = next((b.text for b in resp.content if hasattr(b, "text")), "")
            return {"type": "text", "content": text, "raw": resp}

        else:
            # OpenAI-compatible
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                    },
                }
                for t in tools
            ]
            oai_messages = [{"role": "system", "content": system}] + messages
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=oai_messages,
                tools=openai_tools,
                max_tokens=8192,
            )
            msg = resp.choices[0].message
            if msg.tool_calls:
                tc = msg.tool_calls[0]
                return {
                    "type": "tool_use",
                    "tool_name": tc.function.name,
                    "tool_input": json.loads(tc.function.arguments),
                    "tool_use_id": tc.id,
                    "raw": resp,
                }
            return {"type": "text", "content": msg.content or "", "raw": resp}
