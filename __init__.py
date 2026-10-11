"""Hipocampo hooks plugin — turn-by-turn style learning OUTSIDE the frozen model.

Runs ALONGSIDE any memory.provider (not instead of it): registers
pre_llm_call (whisper context) + post_llm_call (learn from the turn).

State lives per-profile: <hermes_home>/hipocampo/hipocampo.{user}.npz.
User id resolution: sender_id kwarg > memory.hipocampo.user_id config >
HIPPOCAMPO_USER env > "default".
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

try:
    from .hipocampo import Hipocampo, feedback_implicito, set_home_dir
except ImportError:  # loaded as top-level module in some hosts
    from hipocampo import Hipocampo, feedback_implicito, set_home_dir


def _read_config(section: str = "hipocampo") -> dict:
    try:
        from hermes_cli.config import load_config
        mem = load_config().get("memory", {}) or {}
        block = mem.get(section, {}) or {}
        return dict(block) if isinstance(block, dict) else {}
    except Exception:
        return {}


def _resolve_home() -> str:
    """Profile home for .npz files. Prefers HIPPOCAMPO_HOME, else active profile home."""
    env_home = os.getenv("HIPPOCAMPO_HOME", "")
    if env_home:
        try:
            os.makedirs(env_home, exist_ok=True)
            return env_home
        except Exception:
            pass
    try:
        from hermes_constants import get_hermes_home
        home = str(get_hermes_home())
        d = os.path.join(home, "hipocampo")
        os.makedirs(d, exist_ok=True)
        return home
    except Exception:
        return ""


def _resolve_user_id(kwargs: dict) -> str:
    for key in ("sender_id", "author_id", "user_id", "user_id_alt"):
        val = kwargs.get(key)
        if isinstance(val, str) and val.strip():
            safe = "".join(c for c in val.strip().lower() if c.isalnum())
            if safe:
                return safe
    cfg = _read_config()
    uid = cfg.get("user_id", "")
    if isinstance(uid, str) and uid.strip():
        return uid.strip()
    return os.getenv("HIPPOCAMPO_USER", "") or "default"


_instances: dict[tuple[str, str], Hipocampo] = {}


def _get_hp(home: str, user_id: str) -> Hipocampo:
    key = (home, user_id)
    hp = _instances.get(key)
    if hp is None:
        hp = Hipocampo(usuario=user_id, home=home or None)
        _instances[key] = hp
    return hp


def hipocampo_pre_llm_call(session_id: str = "", user_message: str = "",
                           **kwargs) -> dict | None:
    """pre_llm_call: whisper living context into the turn."""
    try:
        home = _resolve_home()
        if home:
            set_home_dir(os.path.join(home, "hipocampo"))
        user_id = _resolve_user_id(kwargs)
        hp = _get_hp(home, user_id)
        ctx_text = hp.susurrar(user_message or "")
        if ctx_text:
            return {"context": ctx_text}
    except Exception as e:
        logger.debug("hipocampo pre_llm_call skipped: %s", e)
    return None


def hipocampo_post_llm_call(session_id: str = "", user_message: str = "",
                            assistant_response: str = "", **kwargs) -> None:
    """post_llm_call: learn from the completed turn."""
    try:
        if not user_message or not assistant_response:
            return
        home = _resolve_home()
        if home:
            set_home_dir(os.path.join(home, "hipocampo"))
        user_id = _resolve_user_id(kwargs)
        hp = _get_hp(home, user_id)
        fb = feedback_implicito(user_message)
        hp.observar(user_message, assistant_response, fb)
    except Exception as e:
        logger.debug("hipocampo post_llm_call skipped: %s", e)


def register(ctx) -> None:
    """Register pre/post LLM hooks (no memory provider — runs alongside any)."""
    ctx.register_hook("pre_llm_call", hipocampo_pre_llm_call)
    ctx.register_hook("post_llm_call", hipocampo_post_llm_call)
