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


def chat(port: int, hf_id: str, messages: list[dict], *, max_tokens: int = 1024,
         timeout: float = 600.0) -> str:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    try:
        r = httpx.post(url, json={"model": hf_id, "messages": messages,
                                  "max_tokens": max_tokens, "temperature": 0},
                       timeout=timeout)
    except httpx.HTTPError as e:
        raise RuntimeError(f"cannot reach model endpoint {url}: {e}") from e
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} from model: {r.text[:500]}")
    msg = r.json()["choices"][0]["message"]
    # reasoning models leave content null and put the text in a reasoning field
    return msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""


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


def run_look(port: int, hf_id: str, content: list[dict], *, max_tokens: int,
             timeout: float) -> str:
    return chat(port, hf_id, [{"role": "user", "content": content}],
                max_tokens=max_tokens, timeout=timeout)


def run_assert(port: int, hf_id: str, content: list[dict], *, max_tokens: int,
               timeout: float) -> dict:
    messages = [{"role": "system", "content": ASSERT_SYSTEM},
                {"role": "user", "content": content}]
    raw = chat(port, hf_id, messages, max_tokens=max_tokens, timeout=timeout)
    try:
        obj = extract_json(raw)
        return {"pass": bool(obj.get("pass")), "reason": str(obj.get("reason", "")),
                "evidence": str(obj.get("evidence", ""))}
    except (ValueError, json.JSONDecodeError) as e:
        return {"pass": False, "reason": f"could not parse model response: {e}",
                "evidence": raw[:500]}
