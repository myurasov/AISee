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
