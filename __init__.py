"""Hipocampo memory plugin — MemoryProvider interface.

Online style adaptation via implicit feedback (length, formality, code vs prose).
Learns user preferences turn-by-turn and whispers them to the frozen cortex.
"""

from __future__ import annotations

from agent.memory_provider import MemoryProvider

try:
    from .hipocampo import Hipocampo, feedback_implicito
except ImportError:
    from hipocampo import Hipocampo, feedback_implicito


def _read_config_user_id() -> str:
    """Lee `memory.hipocampo.user_id` de la config global de Hermes."""
    try:
        from hermes_cli.config import load_config
        mem = load_config().get("memory", {}) or {}
        hip = mem.get("hipocampo", {}) or {}
        uid = hip.get("user_id")
        if isinstance(uid, str) and uid.strip():
            return uid.strip()
    except Exception:
        pass
    return ""


class HipocampoMemoryProvider(MemoryProvider):
    """Hipocampo-style learning memory provider."""

    def __init__(self, config: dict = None):
        config = dict(config) if config else {}
        self.user_id = (
            config.get("user_id")
            or _read_config_user_id()
            or "default"
        )
        self.hp = Hipocampo(usuario=self.user_id)

    @property
    def name(self) -> str:
        return "hipocampo"

    def is_available(self) -> bool:
        return True

    def get_config_schema(self):
        return [
            {"key": "user_id", "description": "User ID for hipocampo namespace", "default": "default", "type": "string"},
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        pass

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return self.hp.susurrar(query)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        pass

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        fb = feedback_implicito(user_content)
        self.hp.observar(user_content, assistant_content, fb)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        pass

    def on_pre_compress(self, messages: list[dict]) -> str:
        return ""

    def get_tool_schemas(self) -> list[dict]:
        return []

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        from tools.registry import tool_error
        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        pass


def register(ctx) -> None:
    """Register Hipocampo as a memory provider plugin."""
    ctx.register_memory_provider(HipocampoMemoryProvider())