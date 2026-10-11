#!/usr/bin/env python3
"""Hipocampo vivo: pesos pequenos que SI cambian en vivo (ms, CPU).

- Huella tematica: vector rapido W (2048 dims) con media movil Hebbiana.
- Estilo: 3 unidades logisticas online (largo/corto, formal/casual, codigo/texto)
  entrenadas por SGD con gradiente logistico real + L2.
- Feedback implicito multi-idioma (ES/EN/PT): gracias/thanks/obrigado=+1,
  mal/bad/ruim=-1, guardalo/remember this=+2.
- Engagement dinamico: mensajes largos del usuario pesan mas.
- Multi-usuario y multi-perfil: un .npz por user_id en <home>/hipocampo/
  (hermes_home que pasa initialize(), o HIPPOCAMPO_HOME, o junto al modulo).
- Todo en numpy, sin GPU, sin parar el servidor.
- Configurable via env vars: HIPPOCAMPO_DIM, HIPPOCAMPO_LR_TEMA, HIPPOCAMPO_LR_ESTILO,
  HIPPOCAMPO_DECAY, HIPPOCAMPO_L2, HIPPOCAMPO_USER, HIPPOCAMPO_HOME.
"""
import hashlib
import json
import os
import re

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))


def _home_dir() -> str:
    """Directorio base para los .npz: HIPPOCAMPO_HOME > junto al modulo."""
    home = os.getenv("HIPPOCAMPO_HOME", "")
    if home and os.path.isdir(home):
        return home
    return BASE


def set_home_dir(path: str) -> str:
    """Fija el directorio base (lo crea si no existe). Devuelve el path usado."""
    global _HOME_OVERRIDE
    _HOME_OVERRIDE = path
    os.makedirs(path, exist_ok=True)
    return path


_HOME_OVERRIDE = ""


def _base_dir() -> str:
    if _HOME_OVERRIDE and os.path.isdir(_HOME_OVERRIDE):
        return _HOME_OVERRIDE
    return _home_dir()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default) or default


def ruta_hp(usuario: str = None, home: str = None) -> str:
    usuario = usuario or _env_str("HIPPOCAMPO_USER", "default")
    seg = "".join(c for c in usuario.lower() if c.isalnum()) or "default"
    base = home or _base_dir()
    d = os.path.join(base, "hipocampo")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        d = base
    return os.path.join(d, f"hipocampo.{seg}.npz")


# ---- Config via env vars (con defaults que coinciden con SKILL.md) ----
DIM = _env_int("HIPPOCAMPO_DIM", 2048)
LR_TEMA = _env_float("HIPPOCAMPO_LR_TEMA", 0.15)
LR_ESTILO = _env_float("HIPPOCAMPO_LR_ESTILO", 0.3)
DECAY_TEMA = _env_float("HIPPOCAMPO_DECAY", 0.995)
L2_ESTILO = _env_float("HIPPOCAMPO_L2", 0.001)
# ----------------------------------------------------------------------

STOP = {"que", "de", "la", "el", "en", "y", "a", "los", "se", "con", "por",
        "una", "para", "es", "mi", "me", "te", "tu", "yo", "esto", "esta",
        "the", "and", "que", "como", "pero", "mas", "este", "hay"}


def _vec(texto):
    v = np.zeros(DIM, dtype=np.float32)
    for w in re.findall(r"\w+", texto.lower()):
        if w in STOP or len(w) < 3:
            continue
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % DIM] += 1.0 if (h >> 7) & 1 else -1.0  # hash con signo
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class Hipocampo:
    def __init__(self, usuario=None, home: str = None):
        self.usuario = usuario or _env_str("HIPPOCAMPO_USER", "default")
        self.path = ruta_hp(self.usuario, home)
        self.W = np.zeros(DIM, dtype=np.float32)      # huella tematica viva
        self.S = np.zeros((3, DIM), dtype=np.float32)  # estilo: largo, formal, codigo
        self.b = np.zeros(3, dtype=np.float32)
        self.turnos = 0
        self.cargar()

    def cargar(self):
        # Migra .npz viejos (junto al modulo o en cwd) a la ruta actual
        for legacy in (
            os.path.join(BASE, "hipocampo.npz"),
            os.path.join(BASE, f"hipocampo.{self.usuario}.npz") if os.path.dirname(self.path) != BASE else None,
            os.path.join(os.getcwd(), f"hipocampo.{self.usuario}.npz"),
        ):
            if legacy and legacy != self.path and not os.path.exists(self.path) and legacy and os.path.exists(legacy):
                try:
                    import shutil
                    os.makedirs(os.path.dirname(self.path), exist_ok=True)
                    shutil.copy(legacy, self.path)  # migracion 1 vez
                    break
                except Exception:
                    pass
        need_backup = False
        try:
            if os.path.exists(self.path):
                with np.load(self.path) as d:
                    if d["W"].shape == (DIM,) and d["S"].shape == (3, DIM):
                        self.W = d["W"].astype(np.float32)
                        self.S = d["S"].astype(np.float32)
                        self.b = d["b"].astype(np.float32)
                        self.turnos = int(d["turnos"])
                    else:
                        need_backup = True
            else:
                need_backup = False
        except Exception:
            need_backup = True
        if need_backup:
            self._backup()

    def _backup(self):
        try:
            if os.path.exists(self.path):
                os.replace(self.path, self.path + ".bak")
        except Exception:
            pass

    def guardar(self):
        np.savez(self.path, W=self.W, S=self.S, b=self.b,
                 turnos=np.array(self.turnos))

    def observar(self, user_msg, asis_msg, feedback=0.0):
        """Actualiza pesos en vivo. Devuelve norma del cambio (prueba de vida)."""
        x = _vec(user_msg + " " + asis_msg[:300])
        antes = self.W.copy()
        # engagement: mensaje largo del usuario = mas peso a este turno
        eng = 0.5 + min(len(user_msg) / 400.0, 1.0)
        # Hebb con olvido controlado
        self.W *= DECAY_TEMA
        self.W += LR_TEMA * eng * (x - self.W)
        # Estilo por SGD logistico con feedback implicito
        if feedback != 0.0:
            largo = 1.0 if len(asis_msg) > 600 else 0.0
            formal = 1.0 if any(w in asis_msg.lower() for w in
                                ["estimado", "cordialmente", "usted"]) else 0.0
            codigo = 1.0 if "```" in asis_msg or "def " in asis_msg else 0.0
            y = np.array([largo, formal, codigo])
            p = _sig(self.S @ x + self.b)
            g = (y - p) * feedback  # gradiente logistico con signo del feedback
            self.S += LR_ESTILO * np.outer(g, x) - L2_ESTILO * self.S
            self.b += LR_ESTILO * g * 0.1
        self.turnos += 1
        self.guardar()
        return float(np.linalg.norm(self.W - antes))

    def susurrar(self, user_msg):
        """Contexto vivo para inyectar al cortex congelado."""
        x = _vec(user_msg)
        afin = float(x @ self.W)  # cuanto resuena el tema con lo vivido
        p = _sig(self.S @ x + self.b)
        estilos = []
        if p[0] > 0.6:
            estilos.append("respuestas desarrolladas")
        elif p[0] < 0.4 and self.turnos > 3:
            estilos.append("respuestas concisas")
        if p[2] > 0.6:
            estilos.append("con ejemplos de codigo")
        txt = (f"[hipocampo vivo turnos={self.turnos} afinidad={afin:.2f}]")
        if estilos:
            txt += " Preferencias detectadas: " + ", ".join(estilos) + "."
        return txt

    def estado(self):
        return {"turnos": self.turnos,
                "norma_W": float(np.linalg.norm(self.W)),
                "norma_S": float(np.linalg.norm(self.S))}


# Feedback implicito multi-idioma (ES / EN / PT / FR / DE / IT).
# Cada idioma aporta sus gracias / quejas / "recuerdalo".
# Se matchea por subcadena en minusculas: "thanks!" o "thank you" valen.
_REMEMBER_CUES = (
    # ES
    "guardalo", "guárdalo", "recuerdalo", "recuérdalo", "no lo olvides",
    # EN
    "remember this", "don't forget", "dont forget", "remember that",
    "save this", "keep this in mind",
    # PT
    "lembre-se", "lembre se", "não se esqueça", "nao se esqueca", "guarde isso",
    # FR
    "rappelle-toi", "rappelle toi", "n'oublie pas", "noublie pas", "retiens ça",
    # DE
    "merk dir das", "vergiss das nicht", "denk daran",
    # IT
    "ricordalo", "non dimenticare", "tienilo a mente",
)

_POSITIVE_CUES = (
    # ES
    "gracias", "perfecto", "genial", "buenisimo", "buenísimo", "excelente",
    "buen trabajo", "bien hecho",
    # EN
    "thanks", "thank you", "perfect", "great", "awesome", "excellent",
    "good job", "well done", "nice work",
    # PT
    "obrigado", "obrigada", "perfeito", "ótimo", "otimo", "excelente",
    "bom trabalho", "bem feito",
    # FR
    "merci", "parfait", "génial", "genial", "excellent", "beau travail",
    # DE
    "danke", "perfekt", "großartig", "grossartig", "ausgezeichnet", "gut gemacht",
    # IT
    "grazie", "perfetto", "fantastico", "eccellente", "ben fatto",
)

_NEGATIVE_CUES = (
    # ES
    "mal", "no me gusta", "horrible", "esta mal", "está mal", "incorrecto",
    "fatal", "pesimo", "pésimo",
    # EN
    "bad", "don't like", "dont like", "horrible", "wrong", "incorrect",
    "awful", "terrible", "that's wrong", "not good",
    # PT
    "ruim", "não gostei", "nao gostei", "horrível", "horrivel", "errado",
    "incorreto", "péssimo", "pessimo",
    # FR
    "mauvais", "pas bon", "horrible", "faux", "incorrect", "nul",
    # DE
    "schlecht", "falsch", "schrecklich", "furchtbar", "inkorrekt",
    # IT
    "male", "sbagliato", "orribile", "scorretto", "pessimo",
)


def feedback_implicito(user_msg):
    t = user_msg.lower()
    if any(w in t for w in _REMEMBER_CUES):
        return 2.0  # confirmacion explicita: aprende fuerte
    if any(w in t for w in _POSITIVE_CUES):
        return 1.0
    if any(w in t for w in _NEGATIVE_CUES):
        return -1.0
    return 0.0
