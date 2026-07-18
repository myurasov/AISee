# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""Post-hoc cleanup of repetitive / degenerate VLM answers.

VLMs narrating sampled frames tend to (a) restate the same observation once per frame
(timestamps varying), (b) oscillate between two contradictory readings, (c) loop a whole
block verbatim (e.g. one email-list row dozens of times), and (d) degenerate into long
single-character runs. All of that defeats answer budgets and buries the signal. This
module collapses those patterns deterministically after generation.

Modes:
- watch chunks: aggressive - sentences are compared with digits folded (per-timestamp
  restatements count as repeats) and any block repeated >= 2x consecutively collapses.
- look: conservative - exact comparison (digits preserved, so table rows and digit runs
  that OCR legitimately repeats survive) and a block must repeat >= 4x to collapse.

Sentence boundaries require whitespace after the punctuation, so decimals ("102.5s")
never split; bare lines (list/table rows) are their own units.
"""

import re

_MAX_BLOCK_UNITS = 12   # longest repeating block (in sentences/lines) we look for
_MAX_CHAR_RUN = 30      # a single character repeated longer than this is degeneration


def _units(text: str) -> list[str]:
    """Sentence/line units, separators attached; decimal points never split."""
    parts = re.split(r"((?<=[.!?])[ \t]+|\n+)", text)
    units: list[str] = []
    for i in range(0, len(parts), 2):
        seg = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
        if seg:
            units.append(seg)
    return units


def _norm(unit: str, fold_digits: bool) -> str:
    u = unit.lower()
    if fold_digits:
        u = re.sub(r"\d+", "#", u)
    return re.sub(r"\s+", " ", u).strip(" \n\t.!?")


def squash_char_runs(text: str) -> tuple[str, int]:
    """Collapse degenerate single-character runs ('9999...' to the token cap)."""
    hits = 0

    def repl(m: re.Match) -> str:
        nonlocal hits
        hits += 1
        return m.group(1) * 10 + f" [degenerate run: same character x{len(m.group(0))}]"

    return re.sub(r"(\S)\1{%d,}" % _MAX_CHAR_RUN, repl, text), hits


def collapse_repeats(text: str, *, fold_digits: bool = True,
                     min_cycles: int = 2) -> tuple[str, int, bool]:
    """(cleaned_text, units_removed, unstable).

    Finds the longest consecutive repetition of any block of 1.._MAX_BLOCK_UNITS units
    and collapses it to a single occurrence plus a note. A two-unit alternation of
    differing statements (A/B/A/B) is contradiction, not information: it is replaced by
    a low-confidence line and flagged unstable.
    """
    text, char_hits = squash_char_runs(text)
    units = _units(text)
    norm = [_norm(u, fold_digits) for u in units]
    removed = char_hits  # count degenerate runs as cleanup work too
    unstable = False
    out: list[str] = []
    i = 0
    while i < len(units):
        best_k, best_r = 0, 1
        for k in range(1, min(_MAX_BLOCK_UNITS, (len(units) - i) // 2) + 1):
            if not any(norm[i:i + k]):
                continue
            r = 1
            while (i + (r + 1) * k <= len(units)
                   and norm[i + r * k:i + (r + 1) * k] == norm[i:i + k]):
                r += 1
            if r >= min_cycles and (r - 1) * k > (best_r - 1) * best_k:
                best_k, best_r = k, r
        if best_k:
            block = units[i:i + best_k]
            removed += (best_r - 1) * best_k
            if best_k == 2 and norm[i] != norm[i + 1]:
                a = block[0].strip().rstrip(".!?\n")
                b = block[1].strip().rstrip(".!?\n")
                out.append(f'The reading alternates between "{a}" and "{b}" across '
                           "sampled frames - low confidence, verify with a still if "
                           "it matters.\n")
                unstable = True
            else:
                out.extend(block)
                note = ("(the preceding observation repeats "
                        if best_k == 1 else
                        f"(the preceding {best_k}-line block repeats ")
                out.append(note + f"{best_r}x consecutively - collapsed.)\n")
            i += best_k * best_r
        else:
            out.append(units[i])
            i += 1
    if removed == 0:
        return text, 0, False
    return "".join(out), removed, unstable


# ---------------- risky-claim extraction (for the watch still cross-check) ----------------

# a quoted string DIRECTLY preceded by a naming cue is a "title claim"; VLMs reproducibly
# invent such titles in video mode, so they get verified against a still. The cue must be
# adjacent - a quoted email sender in a sentence that merely mentions a window is not a
# title claim (learned from a false positive that removed real Outlook content)
_TITLE_CLAIM = re.compile(
    r"\b(?:titled|named|called|labell?ed|headed)\s*[:,]?\s*[\"“''`]([^\"“”''`]{4,90})[\"”''`]",
    re.IGNORECASE)
# share-state claims ("PDF now shared as presentation content") flip readily between
# frames and produce confident false narratives - verified against a still too
_SHARE_CUE = re.compile(r"\b(re-?shared|shared|sharing|presenter|presentation content|"
                        r"presenting)\b", re.IGNORECASE)
_SHARE_OBJ = re.compile(r"\b(pdf|document|screen|window|slide|content|file)\b", re.IGNORECASE)
_TS = re.compile(r"\b(\d+(?:\.\d+)?)\s*s\b")


def extract_risky_claims(text: str, max_claims: int = 2) -> list[dict]:
    """Claims worth verifying against a still frame, each with a timestamp hint (seconds,
    in the same reference frame as the timestamps in the text) when one is present."""
    claims: list[dict] = []
    seen: set[str] = set()
    for unit in _units(text):
        ts = None
        m = _TS.search(unit)
        if m:
            ts = float(m.group(1))
        for q in _TITLE_CLAIM.findall(unit):
            key = q.lower()
            if key not in seen:
                seen.add(key)
                claims.append({"kind": "title", "quote": q,
                               "sentence": unit.strip(), "ts": ts})
        if (_SHARE_CUE.search(unit) and _SHARE_OBJ.search(unit)
                and "not shared" not in unit.lower()
                and "local" not in unit.lower()):
            key = "share:" + _norm(unit, True)
            if key not in seen:
                seen.add(key)
                claims.append({"kind": "share", "quote": None,
                               "sentence": unit.strip(), "ts": ts})
    return claims[:max_claims]


def drop_sentences_mentioning(text: str, needle: str, note: str) -> str:
    """Remove every sentence containing `needle` and append an explanatory note."""
    kept = [u for u in _units(text) if needle.lower() not in u.lower()]
    return "".join(kept).rstrip() + "\n" + note + "\n"


def replace_sentence(text: str, sentence: str, replacement: str) -> str:
    return text.replace(sentence, replacement, 1)
