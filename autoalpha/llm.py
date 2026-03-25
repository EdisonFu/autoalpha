import os
import json
from typing import List, Dict, Any


class CreditExhaustedError(RuntimeError):
    """Raised when the API account has insufficient credits."""


class LLMClient:
    def __init__(self, config: dict = None):
        """
        config 可选。不传时从环境变量自动初始化：
          - ANTHROPIC_API_KEY  → 使用 Anthropic claude-sonnet-4-6
          - DEEPSEEK_API_KEY   → 使用 DeepSeek deepseek-chat
          - OPENAI_API_KEY     → 使用 OpenAI gpt-4o
        """
        if config is None:
            config = self._auto_config()
        self.config = config
        self.model = config["model"]
        provider = config.get("provider", "anthropic")
        api_key = os.environ.get(config.get("api_key_env", "ANTHROPIC_API_KEY"), "")

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
            return {"provider": "anthropic", "model": "claude-sonnet-4-6",
                    "api_key_env": "ANTHROPIC_API_KEY"}
        if os.environ.get("DEEPSEEK_API_KEY"):
            return {"provider": "openai", "model": "deepseek-chat",
                    "api_key_env": "DEEPSEEK_API_KEY",
                    "base_url": "https://api.deepseek.com/v1"}
        if os.environ.get("OPENAI_API_KEY"):
            return {"provider": "openai", "model": "gpt-4o",
                    "api_key_env": "OPENAI_API_KEY"}
        raise EnvironmentError(
            "No API key found. Set one of: ANTHROPIC_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY"
        )

    def complete(self, system: str = "", user: str = "", prompt: str = "",
                 temperature: float = None) -> str:
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
