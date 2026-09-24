"""Tutorial narration: caption parsing, spoken-action detection and alignment with the video.

Captions come from the video platform (YouTube's json3 carries per-word times from speech
recognition; VTT and SRT carry per-cue times). Three things are done with them:

1. **Parsing.** Words with times, grouped into short phrases. YouTube's rolling auto-captions repeat
   every line twice in VTT; the duplicates are removed.
2. **Spoken actions.** Phrases like "press E to extrude" or "pulsamos la G" name Blender
   operations. They are detected with a small English/Spanish lexicon and become *evidence* for
   the visual transitions near them, never steps on their own (a narrator also explains things
   they do not do).
3. **Alignment.** Captions are timed to the audio, but narrators say what they are about to do a
   moment before (or after) doing it. The lag between spoken actions and visual change events is
   estimated by a grid search, and applied only when it beats chance clearly.

All times are seconds on the source video's timeline.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from lucius.errors import MediaError

CAPTION_EXT = {".json3", ".vtt", ".srt"}

PHRASE_GAP_S = 1.0        # a pause this long ends a phrase
PHRASE_MAX_S = 8.0
PHRASE_MAX_WORDS = 18


@dataclass
class Word:
    t: float
    end: float
    text: str


@dataclass
class Cue:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Mention:
    """A spoken reference to a Blender operation."""

    t: float                   # time of the first matched word
    action: str                # Lucius action type
    phrase: str                # the phrase it was said in
    via: str                   # "keyword" or "hotkey"
    params: dict[str, Any] = field(default_factory=dict)


def _normalise(text: str) -> str:
    """Lowercase without accents, so 'extrusión' and 'extrusion' match the same pattern."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


# -- parsing ---------------------------------------------------------------------------------------------

def _parse_json3(data: dict[str, Any]) -> list[Word]:
    words: list[Word] = []
    for event in data.get("events", []):
        segs = event.get("segs")
        if not segs:
            continue
        start = event.get("tStartMs", 0) / 1000.0
        event_end = start + event.get("dDurationMs", 0) / 1000.0
        for seg in segs:
            text = (seg.get("utf8") or "").strip()
            if not text:
                continue
            words.append(Word(t=start + seg.get("tOffsetMs", 0) / 1000.0, end=event_end, text=text))
    words.sort(key=lambda w: w.t)
    # Rolling captions keep each line on screen past the next one: a word ends when the next begins.
    for current, following in zip(words, words[1:]):
        current.end = max(current.t, min(current.end, following.t))
    return words


_TIME = r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})"
_CUE_TIMING = re.compile(_TIME + r"\s*-->\s*" + _TIME)
_INLINE = re.compile(r"<" + _TIME + r">")
_TAG = re.compile(r"</?[^>]+>")


def _seconds(h: str | None, m: str, s: str, ms: str) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _parse_cued(text: str) -> list[Word]:
    """VTT or SRT. Inline word times (YouTube VTT) are used when present; otherwise words are spread
    evenly over their cue. Lines repeated from the previous cue (rolling captions) are dropped."""
    # Line by line: YouTube's VTT puts a whitespace-only line between a cue's timing and its text,
    # so blank lines do not reliably separate cues.
    cues: list[tuple[float, float, list[str]]] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        m = _CUE_TIMING.search(line)
        if m:
            if cues and cues[-1][2] and cues[-1][2][-1].strip().isdigit():
                cues[-1][2].pop()  # an SRT cue number belongs to the next cue
            cues.append((_seconds(*m.groups()[:4]), _seconds(*m.groups()[4:]), []))
        elif cues and line.strip():
            cues[-1][2].append(line)
    words: list[Word] = []
    previous_lines: list[str] = []
    for start, end, body in cues:
        new_lines = [line for line in body if _TAG.sub("", line).strip() not in previous_lines]
        previous_lines = [_TAG.sub("", line).strip() for line in body]
        for line in new_lines:
            if _INLINE.search(line):
                # "word<00:00:01.200><c> next</c><00:00:01.500><c> words</c>"
                pieces = re.split(r"(<" + _TIME + r">)", line)
                t = start
                index = 0
                while index < len(pieces):
                    piece = pieces[index]
                    stamp = _INLINE.fullmatch(piece or "")
                    if stamp:
                        t = _seconds(*stamp.groups())
                        index += 5  # the split also returns the four captured groups
                        continue
                    for token in _TAG.sub("", piece or "").split():
                        words.append(Word(t=t, end=end, text=token))
                    index += 1
            else:
                tokens = _TAG.sub("", line).split()
                span = max(0.001, end - start)
                for i, token in enumerate(tokens):
                    words.append(Word(t=start + span * i / max(1, len(tokens)), end=end, text=token))
    words.sort(key=lambda w: w.t)
    for current, following in zip(words, words[1:]):
        current.end = max(current.t, min(current.end, following.t))
    return words


def _phrases(words: list[Word]) -> list[Cue]:
    cues: list[Cue] = []
    current: list[Word] = []
    for word in words:
        if current and (word.t - current[-1].end > PHRASE_GAP_S or word.t - current[0].t > PHRASE_MAX_S
                        or len(current) >= PHRASE_MAX_WORDS):
            cues.append(Cue(current[0].t, current[-1].end, " ".join(w.text for w in current), current))
            current = []
        current.append(word)
        if word.text.endswith((".", "?", "!")):
            cues.append(Cue(current[0].t, current[-1].end, " ".join(w.text for w in current), current))
            current = []
    if current:
        cues.append(Cue(current[0].t, current[-1].end, " ".join(w.text for w in current), current))
    return cues


class CaptionTrack:
    def __init__(self, words: list[Word], *, language: str | None = None, source: str = "unknown") -> None:
        self.words = words
        self.language = language
        self.source = source            # "auto" (speech recognition), "manual" or "unknown"
        self.cues = _phrases(words)

    @classmethod
    def from_file(cls, path: str | Path, *, language: str | None = None, source: str | None = None) -> CaptionTrack:
        path = Path(path)
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise MediaError(f"cannot read captions {path.name}: {exc}") from exc
        suffix = path.suffix.lower()
        if suffix == ".json3" or raw.lstrip().startswith("{"):
            try:
                words = _parse_json3(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise MediaError(f"captions {path.name} are not valid json3") from exc
        elif suffix in (".vtt", ".srt") or "-->" in raw:
            words = _parse_cued(raw)
        else:
            raise MediaError(f"unsupported caption format: {path.name}")
        if not words:
            raise MediaError(f"captions {path.name} contain no text")
        if source is None and '"acAsrConf"' in raw:
            source = "auto"  # YouTube marks speech-recognition words with a confidence field
        if language is None:
            # yt-dlp names files <id>.<lang>.<ext>; "-orig" marks the original (untranslated) track.
            parts = path.name.split(".")
            language = parts[-2].removesuffix("-orig") if len(parts) >= 3 else None
        return cls(words, language=language, source=source or "unknown")

    @property
    def span(self) -> tuple[float, float]:
        return (self.words[0].t, self.words[-1].end) if self.words else (0.0, 0.0)

    def window(self, t0: float, t1: float) -> CaptionTrack:
        return CaptionTrack([w for w in self.words if t0 <= w.t < t1], language=self.language, source=self.source)

    def text(self, t0: float, t1: float) -> str:
        return " ".join(w.text for w in self.words if t0 <= w.t < t1)

    def mentions(self) -> list[Mention]:
        return find_mentions(self.cues)

    def to_dict(self) -> dict[str, Any]:
        return {"language": self.language, "source": self.source, "words": [asdict(w) for w in self.words]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaptionTrack:
        return cls([Word(**w) for w in data.get("words", [])], language=data.get("language"),
                   source=data.get("source", "unknown"))


# -- spoken actions --------------------------------------------------------------------------------------

PRIMITIVES = {"cube": "cube", "cubo": "cube", "cylinder": "cylinder", "cilindro": "cylinder", "plane": "plane",
              "plano": "plane", "sphere": "uv_sphere", "esfera": "uv_sphere", "cone": "cone", "cono": "cone",
              "torus": "torus", "toroide": "torus", "dona": "torus", "icosphere": "ico_sphere",
              "icoesfera": "ico_sphere", "circle": "circle", "circulo": "circle"}

# (pattern on accent-free lowercase text, action). Order matters: the first match in a phrase wins
# for overlapping spans, so specific patterns come before general ones.
ACTION_LEXICON: list[tuple[str, str]] = [
    (r"\b(?:add|adding|insert|anad\w*|agreg\w*|insert\w*|cre(?:a|ar|amos))\s+(?:a |an |un |una |el |la )?"
     r"(?:new |nuevo |nueva )?(cube|cylinder|plane|sphere|cone|torus|icosphere|circle|cubo|cilindro|plano|esfera|"
     r"cono|toroide|dona|icoesfera|circulo)\b", "add_primitive"),
    (r"\bloop ?cut|\bcortes? (?:de|en) (?:bucle|loop)|\bcorte en bucle", "loop_cut"),
    (r"\bextru\w*", "extrude"),
    (r"\bbevel\w*|\bbisel\w*|\bchaflan\w*", "bevel"),
    (r"\binset\w*|\binsert\w* (?:de )?caras?|\binset faces?", "inset"),
    (r"\bsubdivi\w*", "subdivide"),
    (r"\bknife\b|\bcuchillo\b", "knife"),
    (r"\bmerge\w*|\bfusion\w*|\bunir (?:los )?vertices", "merge"),
    (r"\bbridge (?:edge )?loops?|\bpuente entre", "bridge_loops"),
    (r"\bduplicat\w*|\bduplic\w*", "duplicate"),
    (r"\bmirror\w*|\bespejo\b|\bsimetri\w*|\bsymmetr\w*", "set_symmetry"),
    (r"\b(?:solidify|array|subdivision surface|subsurf|boolean|booleano|decimate|diezmar)\b", "add_modifier"),
    (r"\b(?:add|anad\w*|agreg\w*|pon\w*)\s+(?:a |an |un |el )?(?:modifier|modificador)", "add_modifier"),
    (r"\bapply (?:the )?(?:scale|rotation|transforms?)|\baplic\w* (?:la |el )?(?:escala|rotacion|transform\w*)",
     "apply_transform"),
    (r"\bapply (?:the )?modifier|\baplic\w* (?:el )?modificador", "apply_modifier"),
    (r"\bshade smooth|\bsombreado suave|\bsuavizad\w* (?:el )?sombreado", "shade_smooth"),
    (r"\bshade flat|\bsombreado plano", "shade_flat"),
    (r"\bedit mode\b|\bmodo (?:de )?edicion\b|\bmodo edit\b", "mode_change"),
    (r"\bobject mode\b|\bmodo objeto\b", "mode_change"),
    (r"\bsculpt mode\b|\bmodo escultura\b", "mode_change"),
    (r"\b(?:front|side|top|right|left|back) view\b|\bvista (?:frontal|lateral|superior|de arriba|derecha|izquierda)",
     "view_preset"),
    (r"\bscal(?:e|ing|ed)\b|\bescal(?:a|ar|amos|ado|emos)\b|\bresiz\w*", "scale"),
    (r"\brotat\w*|\brot(?:ar|amos|a|emos)\b|\bgir(?:ar|amos|a)\b", "rotate"),
    (r"\bmove\b|\bmoving\b|\bgrab\w*|\btranslat\w*|\bmov(?:er|emos|emos)\b|\bmueve\w*|\bdesplaz\w*", "translate"),
    (r"\bdelete\w*|\bborr(?:ar|amos|a)\b|\belimin\w*|\bsuprim\w*", "delete"),
    (r"\bselect all\b|\bseleccion\w* todo", "select_all"),
]

_MODIFIER_TYPES = {"solidify": "SOLIDIFY", "array": "ARRAY", "subdivision surface": "SUBSURF", "subsurf": "SUBSURF",
                   "boolean": "BOOLEAN", "booleano": "BOOLEAN", "decimate": "DECIMATE", "diezmar": "DECIMATE",
                   "mirror": "MIRROR", "espejo": "MIRROR"}

HOTKEY_ACTIONS = {
    "e": "extrude", "g": "translate", "r": "rotate", "s": "scale", "i": "inset", "k": "knife", "x": "delete",
    "m": "merge", "f": "fill", "a": "select_all", "tab": "mode_change", "ctrl+r": "loop_cut", "ctrl+b": "bevel",
    "shift+d": "duplicate", "shift+a": "add_menu", "ctrl+a": "apply_transform", "alt+a": "deselect_all",
    "f3": "search_menu",
}
_MODS = {"ctrl": "ctrl", "control": "ctrl", "shift": "shift", "mayus": "shift", "mayusculas": "shift", "alt": "alt"}
# "press E", "hit ctrl + R", "pulsamos la G", "presiona control R", "con la tecla S", "shift D",
# "le damos a la E", "le doy a la S", "dale a la G" (Spanish "darle a" = hit)
_HOTKEY = re.compile(
    r"\b(?:press(?:ing)?|hit(?:ting)?|pulsa\w*|presion\w*|aprieta\w*|tecla|letra|shortcut|atajo|"
    r"(?:doy|das|da|damos|dale|darle|dandole)\s+a(?=\s+(?:la|el|tecla|letra|control|ctrl|shift|mayus|alt)\b))\s+"
    r"(?:the |la |el |tecla |letra |key )*"
    r"(?:(ctrl|control|shift|mayus|mayusculas|alt)\s*(?:\+|plus|mas|y)?\s*)?"
    r"(f3|tab|tabulador|[a-z])\b"
    r"|\b(ctrl|control|shift|mayus|alt)\s*(?:\+|plus|mas)?\s*([a-z])\b")


def find_mentions(cues: list[Cue]) -> list[Mention]:
    mentions: list[Mention] = []
    for cue in cues:
        if not cue.words:
            continue
        # Character offset of every word in the normalised phrase, to time each match by its first word.
        offsets, pieces, position = [], [], 0
        for word in cue.words:
            normalised = _normalise(word.text)
            offsets.append(position)
            pieces.append(normalised)
            position += len(normalised) + 1
        text = " ".join(pieces)

        def time_at(char: int) -> float:
            index = max(0, int(np.searchsorted(offsets, char, side="right")) - 1)
            return cue.words[index].t

        taken: list[tuple[int, int]] = []
        for match in _HOTKEY.finditer(text):
            modifier = match.group(1) or match.group(3)
            key = match.group(2) or match.group(4)
            key = "tab" if key == "tabulador" else key
            combo = f"{_MODS[modifier]}+{key}" if modifier else key
            action = HOTKEY_ACTIONS.get(combo)
            if action is None:
                continue
            if not modifier and key in ("a", "o", "y") and not re.search(
                    r"\b(?:the|la|tecla|letra|key)\s+" + key + r"$", match.group(0)) or (
                    not modifier and match.group(0).split()[0] in ("doy", "das", "da", "damos", "dale", "darle",
                                                                    "dandole") and " la " not in match.group(0)
                    and "tecla" not in match.group(0)):
                continue  # "pulsa a la derecha": a, o and y are ordinary Spanish words unless named as a key
            mentions.append(Mention(t=time_at(match.start()), action=action, phrase=cue.text, via="hotkey",
                                    params={"hotkey": combo}))
            taken.append(match.span())
        for pattern, action in ACTION_LEXICON:
            for match in re.finditer(pattern, text):
                if any(a <= match.start() < b for a, b in taken):
                    continue
                params: dict[str, Any] = {}
                if action == "add_primitive" and match.lastindex:
                    params["kind"] = PRIMITIVES.get(match.group(1), match.group(1))
                if action in ("add_modifier", "set_symmetry"):
                    found = next((v for k, v in _MODIFIER_TYPES.items() if k in match.group(0)), None)
                    if found:
                        params["type"] = found
                mentions.append(Mention(t=time_at(match.start()), action=action, phrase=cue.text, via="keyword",
                                        params=params))
                taken.append(match.span())
    mentions.sort(key=lambda m: m.t)
    return _fold_axes_and_duplicates(mentions)


TRANSFORMS = {"translate", "rotate", "scale", "extrude"}
_AXIS = re.compile(r"\b([xyz])[ -]?(?:axis|direction|direccion)\b|\beje ([xyz])\b")


def _fold_axes_and_duplicates(mentions: list[Mention]) -> list[Mention]:
    """"Press R then X": after a transform key, X/Y/Z constrain the axis (X is not delete there).
    The same action named twice in one breath ("rotate ... press R") counts once."""
    out: list[Mention] = []
    for mention in mentions:
        previous = next((m for m in reversed(out) if mention.t - m.t <= 4.0), None) if out else None
        if (mention.via == "hotkey" and mention.params.get("hotkey") in ("x", "y", "z") and previous is not None
                and previous.action in TRANSFORMS):
            previous.params.setdefault("axis", mention.params["hotkey"])
            continue
        if mention.action in TRANSFORMS and "axis" not in mention.params:
            axis = _AXIS.search(_normalise(mention.phrase))
            if axis:
                mention.params["axis"] = axis.group(1) or axis.group(2)
        duplicate = next((m for m in reversed(out) if mention.t - m.t <= 2.0 and m.action == mention.action), None)
        if duplicate is not None:
            if mention.via == "hotkey" and duplicate.via == "keyword":
                duplicate.via, duplicate.params = "hotkey", {**duplicate.params, **mention.params}
            continue
        out.append(mention)
    return out


# -- alignment -------------------------------------------------------------------------------------------

@dataclass
class LagEstimate:
    """How far visual changes trail the words that announce them (positive: the action comes after)."""

    lag_s: float
    applied: bool
    matched: int
    mentions: int
    events: int
    null_mean: float
    null_std: float
    z: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def estimate_lag(mention_times: list[float], event_times: list[float], *, max_lag_s: float = 6.0,
                 step_s: float = 0.25, tolerance_s: float = 1.0, min_matches: int = 5,
                 min_z: float = 2.0) -> LagEstimate:
    """Grid-search the lag that puts the most spoken actions within ``tolerance_s`` of a visual change.

    Chance level is measured by rotating the mentions around the analysed span by 30 s or more,
    where no real relation remains but the density of events is the same. The lag is applied only when the best score is
    ``min_z`` standard deviations above chance and matches at least ``min_matches`` mentions.
    """
    mentions = np.asarray(sorted(mention_times), dtype=float)
    events = np.asarray(sorted(event_times), dtype=float)
    if len(mentions) == 0 or len(events) == 0:
        return LagEstimate(0.0, False, 0, len(mentions), len(events), 0.0, 0.0, 0.0, "no_mentions_or_events")

    def score(points: np.ndarray) -> int:
        index = np.searchsorted(events, points)
        left = events[np.clip(index - 1, 0, len(events) - 1)]
        right = events[np.clip(index, 0, len(events) - 1)]
        return int(np.sum(np.minimum(np.abs(points - left), np.abs(points - right)) <= tolerance_s))

    lags = np.arange(-max_lag_s, max_lag_s + 1e-9, step_s)
    scores = np.array([score(mentions + lag) for lag in lags])
    best = int(np.argmax(scores))
    # Ties: prefer the smallest absolute lag (no evidence for a larger shift).
    best_scores = np.flatnonzero(scores == scores[best])
    best = int(best_scores[np.argmin(np.abs(lags[best_scores]))])
    # Chance level: rotate the mentions around the analysed span (so none falls outside it) by offsets
    # far larger than any plausible lag.
    start = min(mentions[0], events[0]) - tolerance_s
    span = max(mentions[-1], events[-1]) + tolerance_s - start
    offsets = [o for o in np.arange(30.0, 600.0, 7.3) if o < span - 30.0] or [span / 2]
    null = np.array([score(start + np.mod(mentions - start + o, span)) for o in offsets])
    null_mean, null_std = float(null.mean()), float(max(null.std(), 1.0))
    z = (float(scores[best]) - null_mean) / null_std
    applied = bool(scores[best] >= min_matches and z >= min_z)
    reason = "significant" if applied else ("too_few_matches" if scores[best] < min_matches else "not_above_chance")
    return LagEstimate(lag_s=float(lags[best]) if applied else 0.0, applied=applied, matched=int(scores[best]),
                       mentions=len(mentions), events=len(events), null_mean=null_mean, null_std=null_std, z=z,
                       reason=reason)
