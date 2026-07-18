# Copyright (c) 2026 Mikhail Yurasov <me@yurasov.me>
# SPDX-License-Identifier: Apache-2.0

"""Inference calls against a model container's OpenAI-compatible endpoint."""

import json
import re

import httpx

ASSERT_SYSTEM = (
    "You are a meticulous visual QA inspector. You are shown one or more screenshots (or "
    "sampled video frames) of an application UI. Judge the user's expectation strictly against "
    "what is actually visible. Respond with ONLY a single JSON object, no prose, no code fence: "
    '{"pass": <true|false>, "reason": "<one or two sentences>", "evidence": "<concrete details '
    'you saw, e.g. labels/text/colors/positions>"}. Set pass=false if anything in the expectation '
    "is missing, wrong, or not clearly visible."
)


# vLLM's context-overflow 400: "...maximum context length is X tokens. However, you
# requested Y tokens (Z in the messages, W in the completion)..." (wording varies slightly)
_CTX_OVERFLOW_RE = re.compile(
    r"maximum context length is (\d+) tokens.*?(\d+)\s+(?:tokens?\s+)?in the messages",
    re.DOTALL | re.IGNORECASE)


def chat(port: int, hf_id: str, messages: list[dict], *, max_tokens: int = 1024,
         timeout: float = 600.0) -> tuple[str, dict]:
    """One chat completion. Returns (text, meta) where meta carries finish_reason,
    completion_tokens, and max_tokens_clamped (set when the requested answer budget had
    to be shrunk to fit the context next to a large prompt)."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    clamped = False
    for attempt in (0, 1):
        try:
            r = httpx.post(url, json={"model": hf_id, "messages": messages,
                                      "max_tokens": max_tokens, "temperature": 0},
                           timeout=timeout)
        except httpx.HTTPError as e:
            raise RuntimeError(f"cannot reach model endpoint {url}: {e}") from e
        if r.status_code == 400 and attempt == 0:
            # answer budget + prompt overflow the context: vLLM reports the exact prompt
            # size, so clamp max_tokens to what actually fits and retry once
            m = _CTX_OVERFLOW_RE.search(r.text)
            if m:
                ctx, prompt_tokens = int(m.group(1)), int(m.group(2))
                fitting = ctx - prompt_tokens - 16
                if fitting >= 256 and fitting < max_tokens:
                    max_tokens, clamped = fitting, True
                    continue
                raise RuntimeError(
                    f"prompt ({prompt_tokens} tokens) leaves no room for an answer in the "
                    f"{ctx}-token context - send less media or lower its resolution")
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code} from model: {r.text[:500]}")
        body = r.json()
        choice = body["choices"][0]
        msg = choice["message"]
        usage = body.get("usage") or {}
        meta = {"finish_reason": choice.get("finish_reason"),
                "completion_tokens": usage.get("completion_tokens"),
                "max_tokens": max_tokens, "max_tokens_clamped": clamped}
        # reasoning models leave content null and put the text in a reasoning field
        text = (msg.get("content") or msg.get("reasoning_content")
                or msg.get("reasoning") or "")
        return text, meta
    raise RuntimeError("unreachable")  # the loop always returns or raises


def extract_json(text: str) -> dict:
    """First balanced {...} object in a model response (strips <think> blocks)."""
    if not text:
        raise ValueError("empty model response (reasoning may have consumed the token budget)")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError(f"unbalanced JSON in response: {text[:200]!r}")


def with_context(text: str, context: str | None) -> str:
    if context:
        return f"Context (background provided by the caller):\n{context}\n\n{text}"
    return text


def truncation_marker(meta: dict) -> str:
    n = meta.get("completion_tokens") or meta.get("max_tokens")
    return f" [truncated at {n} tokens]"


def annotate(result: dict, meta: dict) -> dict:
    """Fold chat meta into a task/chunk result: truncated + max_tokens_clamped flags."""
    if meta.get("finish_reason") == "length":
        result["truncated"] = True
    if meta.get("max_tokens_clamped"):
        result["max_tokens_clamped"] = True
    return result


def run_look(port: int, hf_id: str, content: list[dict], *, max_tokens: int,
             timeout: float) -> tuple[str, dict]:
    return chat(port, hf_id, [{"role": "user", "content": content}],
                max_tokens=max_tokens, timeout=timeout)


def run_assert(port: int, hf_id: str, content: list[dict], *, max_tokens: int,
               timeout: float) -> dict:
    messages = [{"role": "system", "content": ASSERT_SYSTEM},
                {"role": "user", "content": content}]
    raw, meta = chat(port, hf_id, messages, max_tokens=max_tokens, timeout=timeout)
    if meta.get("finish_reason") == "length":
        # a clipped verdict is unparseable JSON, not a judgment - fail it distinctly
        return annotate({"pass": False,
                         "reason": f"verdict truncated at "
                                   f"{meta.get('completion_tokens') or max_tokens} tokens "
                                   "- raise max_tokens (reasoning models think inside the "
                                   "same budget)",
                         "evidence": raw[-500:]}, meta)
    try:
        obj = extract_json(raw)
        return annotate({"pass": bool(obj.get("pass")),
                         "reason": str(obj.get("reason", "")),
                         "evidence": str(obj.get("evidence", ""))}, meta)
    except (ValueError, json.JSONDecodeError) as e:
        return annotate({"pass": False, "reason": f"could not parse model response: {e}",
                         "evidence": raw[:500]}, meta)
