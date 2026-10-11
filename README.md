# Hippocampus — Living Procedural Memory for Hermes Agent

**Learns *how* the user likes responses** (long/short, formal/casual, code/prose) turn by turn, without retraining the model.

## What it does

- **Topic fingerprint (W, 2048-d):** Hebbian moving average with controlled decay (`DECAY_TEMA=0.995`). Semantic resonance via signed MD5 hashing.
- **Learned style (S 3x2048 + bias):** Three online logistic units (long, formal, code) trained by SGD with true logistic gradient + L2.
- **Implicit feedback:** `thanks/perfect/great → +1`, `bad/don't like/wrong → -1`, explicit remember cues → `+2`.
- **Dynamic engagement:** Longer user messages weigh more.
- **Multi-user:** One `.npz` per `user_id` (`hipocampo.{user}.npz`).
- **`prefetch()` → whisper:** Each turn injects `[hipocampo vivo turnos=N afinidad=X.XX] ...` into context.
- **`sync_turn()` → learns:** After each response, updates weights (microseconds, CPU, numpy).

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
hermes config set memory.provider hipocampo
hermes config set memory.hipocampo.user_id <your-name>
```

Or in `config.yaml`:
```yaml
memory:
  provider: hipocampo
  hipocampo:
    user_id: "your-name"
```

Restart Hermes. The log should show:
```
Memory provider 'hipocampo' registered (0 tools)
Memory provider 'hipocampo' activated
```

## Requirements

- Python 3.10+
- `numpy` (already in the Hermes venv)

## Configuration (optional, env vars)

| Variable | Default |
|---|---|
| `HIPPOCAMPO_DIM` | 2048 |
| `HIPPOCAMPO_LR_TEMA` | 0.15 |
| `HIPPOCAMPO_LR_ESTILO` | 0.3 |
| `HIPPOCAMPO_DECAY` | 0.995 |
| `HIPPOCAMPO_L2` | 0.001 |
| `HIPPOCAMPO_USER` | default |

## Files

| File | What it is |
|---|---|
| `plugin.yaml` | Manifest (`name: hipocampo`, `kind: exclusive`) |
| `__init__.py` | `HipocampoMemoryProvider` + `register(ctx)` |
| `hipocampo.py` | Core: `Hipocampo` + `feedback_implicito` + env vars |

## Does not replace MEMORY.md

It's **complementary**: `MEMORY.md` stores *facts* (who you are, what you like), the hippocampus learns *style* (how you like responses). Facts vs. style.

## Author

Jose (junior69recpt) + Anita — MIT
