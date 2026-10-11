# Hippocampus — Living Style Memory for Hermes Agent (v1.1)

**Learns *how* the user likes responses** (long/short, formal/casual, code/prose) turn by turn, without retraining the model.

v1.1 runs as **`pre_llm_call` + `post_llm_call` hooks alongside any `memory.provider`** — it no longer occupies the single provider slot, so it works together with mem0 / honcho / byterover / builtin.

## What it does

- **Topic fingerprint (W, 2048-d):** Hebbian moving average with controlled decay (`DECAY_TEMA=0.995`). Semantic resonance via signed MD5 hashing.
- **Learned style (S 3x2048 + bias):** Three online logistic units (long, formal, code) trained by SGD with true logistic gradient + L2.
- **Implicit feedback, 6 languages (ES/EN/PT/FR/DE/IT):** thanks/gracias/obrigado/merci/danke/grazie → `+1`, bad/mal/ruim/mauvais/schlecht/male → `-1`, remember-this/guárdalo/lembre-se/rappelle-toi/merk-dir-das/ricordalo → `+2`.
- **Dynamic engagement:** Longer user messages weigh more.
- **Multi-user + multi-profile:** One `.npz` per `user_id` under `<hermes_home>/hipocampo/` (resolved per turn, legacy auto-migrated).
- **`pre_llm_call` → whisper:** Each turn injects `[hipocampo vivo turnos=N afinidad=X.XX] ...` as hook context.
- **`post_llm_call` → learn:** After each response, updates weights (microseconds, CPU, numpy).

## Installation

```bash
# Copy this folder to your Hermes profile:
#   Windows: %LOCALAPPDATA%\hermes\plugins\hipocampo\
#   Linux/macOS: ~/.hermes/plugins/hipocampo/

# Or install from Git:
hermes plugins install <repo-url>
```

## Activation

```bash
hermes plugins enable hipocampo
```

Keep your existing memory backend untouched — hippocampus runs beside it:

```yaml
# memory.provider stays whatever you already use (or default):
memory:
  provider: default
# Optional: pick the style namespace (default: "default")
# memory:
#   hipocampo:
#     user_id: "your-name"
```

Restart Hermes (or wait for the gateway live-reload). Logs should show the plugin loading with hooks; each turn then carries `[hipocampo vivo turnos=N ...]` context.

## Requirements

- Python 3.10+
- `numpy` (already in the Hermes venv)

## Configuration (optional, env vars)

| Variable | Default | What |
|---|---|---|
| `HIPPOCAMPO_DIM` | 2048 | Topic vector width |
| `HIPPOCAMPO_LR_TEMA` | 0.15 | Hebbian learning rate |
| `HIPPOCAMPO_LR_ESTILO` | 0.3 | Style SGD rate |
| `HIPPOCAMPO_DECAY` | 0.995 | Topic forgetting |
| `HIPPOCAMPO_L2` | 0.001 | Style weight decay |
| `HIPPOCAMPO_USER` | default | Style namespace fallback |
| `HIPPOCAMPO_HOME` | (profile home) | Override state dir |

`user_id` resolution per turn: hook `sender_id` → `memory.hipocampo.user_id` → `HIPPOCAMPO_USER` → `"default"`.

## Files

| File | What it is |
|---|---|
| `plugin.yaml` | Manifest (standalone hooks plugin, v1.1.0) |
| `__init__.py` | `register(ctx)` → `pre_llm_call` + `post_llm_call` |
| `hipocampo.py` | Core: `Hipocampo` + `feedback_implicito` + env vars |

## Does not replace MEMORY.md

It's **complementary**: `MEMORY.md` stores *facts* (who you are, what you like), the hippocampus learns *style* (how you like responses). Facts vs. style.

## Changelog

- **v1.1.0** — Hooks instead of provider (runs alongside any `memory.provider`); 6-language feedback (ES/EN/PT/FR/DE/IT); per-profile state dir (`<hermes_home>/hipocampo/`, legacy auto-migrated).
- **v1.0.0** — Initial `MemoryProvider` release (single-slot, ES-only feedback, state beside plugin).

## Author

Jose (junior69recpt) + Anita — MIT
