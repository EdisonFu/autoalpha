import os
import json
from typing import List, Dict, Any


class CreditExhaustedError(RuntimeError):
    """Raised when the API account has insufficient credits."""


# ── Token 计量 ───────────────────────────────────────────────────────────────
# 价格（美元 / 百万 token）。用于把 token 数换算成实际花费。
# 只列本项目会用到的模型；未知模型按 0 计价并在汇总里标注。
_PRICING = {
    }


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
        pin, pout = _PRICING.get(m, (0.0, 0.0))
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
            "priced": m in _PRICING,
            "by_purpose": dict(self.by_purpose),
        }


class LLMClient:
    def __init__(self, config: dict = None):
        """
        config 可选。不传时从环境变量自动初始化：
          - ANTHROPIC_API_KEY  → 使用 Anthropic <中等模型>
          - DEEPSEEK_API_KEY   → 使用 DeepSeek <中等模型>
          - OPENAI_API_KEY     → 使用 OpenAI <中等模型>
        """
        if config is None:
            config = self._auto_config()
        self.config = config
        self.model = config["model"]
        provider = config.get("provider", "anthropic")
        api_key = os.environ.get(config.get("api_key_env", "ANTHROPIC_API_KEY"), "")

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
        if os.environ.get("ANTHROPIC_API_KEY"):
            return {"provider": "anthropic", "model": "<中等模型>",
                    "api_key_env": "ANTHROPIC_API_KEY"}
        if os.environ.get("DEEPSEEK_API_KEY"):
            return {"provider": "openai", "model": "<中等模型>",
                    "api_key_env": "DEEPSEEK_API_KEY",
                    "base_url": "https://api.deepseek.com/v1"}
        if os.environ.get("OPENAI_API_KEY"):
            return {"provider": "openai", "model": "<中等模型>",
                    "api_key_env": "OPENAI_API_KEY"}
        raise EnvironmentError(
            "No API key found. Set one of: ANTHROPIC_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY"
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
