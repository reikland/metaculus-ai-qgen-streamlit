#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import time
import textwrap
import random
import re
from typing import Dict, Any, List, Optional

import requests
import pandas as pd
import streamlit as st
import json as _json

# ============================================================
# 1. CONFIG
# ============================================================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL_ENV = os.environ.get("OPENROUTER_MODEL", "").strip()

REFERER = "https://localhost"
TITLE = "Metaculus – Proto Question Cluster + Judge + Agent (simple agent)"

DEFAULT_MODEL = "openai/gpt-4o-mini"

if "qgen_result" not in st.session_state:
    st.session_state["qgen_result"] = None


# ============================================================
# 2. OPENROUTER HELPERS
# ============================================================

def get_openrouter_key() -> str:
    try:
        v = st.session_state.get("OPENROUTER_API_KEY_OVERRIDE", "").strip()
    except Exception:
        v = ""

    if not v:
        v = os.environ.get("OPENROUTER_API_KEY", "").strip()

    if not v:
        try:
            if "OPENROUTER_API_KEY" in st.secrets:
                v = str(st.secrets["OPENROUTER_API_KEY"]).strip()
        except Exception:
            pass

    return v


def ascii_safe(s: str) -> str:
    try:
        return s.encode("latin-1", "ignore").decode("latin-1")
    except Exception:
        return "".join(ch for ch in s if ord(ch) < 256)


def or_headers() -> Dict[str, str]:
    key = get_openrouter_key()
    if not key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Referer": ascii_safe(REFERER),
        "X-Title": ascii_safe(TITLE),
        "User-Agent": ascii_safe("metaculus-proto-qgen-judge-agent/0.5"),
    }


def call_openrouter_raw(
    messages: List[Dict[str, str]],
    model: str,
    max_tokens: int = 2000,
    temperature: float = 0.4,
    retries: int = 3,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 1,
        "max_tokens": max_tokens,
    }

    last_error: Optional[Exception] = None

    for k in range(retries):
        try:
            r = requests.post(
                OPENROUTER_URL,
                headers=or_headers(),
                json=payload,
                timeout=120,
            )
            if r.status_code == 429:
                retry_after = float(r.headers.get("Retry-After", "2") or 2)
                time.sleep(min(retry_after, 10))
                continue

            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(str(data["error"]))

            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError("No choices in OpenRouter response")

            content = choices[0].get("message", {}).get("content", "")
            return content or ""
        except Exception as e:
            last_error = e
            time.sleep(0.8 * (k + 1))

    raise RuntimeError(f"[openrouter] giving up after retries: {repr(last_error)}")


# ============================================================
# 3. PROMPTS – GENERATOR / JUDGE / SIMPLE AGENT
# ============================================================

# ---------------------- Proto cluster generator ----------------------

GEN_SYS_PROTO = """
You generate CLUSTERS of proto forecasting questions for Metaculus.

You are used inside an automated pipeline which will PARSE your output.
If you deviate from the required format, your entire answer is discarded as unusable.

ABSOLUTE RULES
- You MUST strictly follow the template described below.
- You MUST produce EXACTLY N questions: not fewer, not more.
- You MUST NOT add explanations, apologies, headings, or commentary.
- Your VERY FIRST non-empty line MUST be: "QUESTION 1".
- Your LAST non-empty line MUST start with "Candidate-source:" for QUESTION N.
- No markdown, no bullet points, no JSON, no code fences.

CLUSTER BEHAVIOUR
- Interpret the seed prompt as describing ONE central theme (the "cluster theme").
- Produce a coherent cluster of N related proto-questions.
- 1–2 questions should be broad "anchor" questions about the central theme (Role=CORE).
- All remaining questions should be narrower "variants" exploring different angles of the same theme (Role=VARIANT).

Content constraints:
- Questions must be about an uncertain future or as-yet-unobserved outcome.
- They must be potentially resolvable from public data (official statistics, major datasets, government reports, reputable newswires).
- Avoid trivial questions whose probability is obviously ~0% or ~100%.
- Avoid questions that are already resolved.

FORMAT (STRICT, LINE-BASED)
For each i = 1..N you output a block with these 5 lines:

QUESTION i
Role: CORE or VARIANT
Title: <short title, <= 100 characters, single line>
Question: <1–3 sentences, single line, ends with "?" or equivalent>
Angle: <short phrase for the angle within the cluster>
Candidate-source: <one likely family of public resolution sources or datasets, single line>

No blank lines inside a block are required, but they are allowed between blocks.
NEVER output any example or meta-commentary. Only the blocks above.
""".strip()

GEN_USER_TMPL_PROTO = textwrap.dedent(
    """
    You will now generate a CLUSTER of proto forecasting questions.

    HARD CONSTRAINT:
    - N_questions = {n}.
    - You MUST output EXACTLY N_questions blocks, labelled QUESTION 1, QUESTION 2, ..., QUESTION {n}.
    - If you output fewer or more questions, or any text outside the template, the output is considered INVALID.

    Seed prompt (1–12 sentences, central theme):
    {seed}

    Optional context:
    - Domain tags: {tags}
    - Horizon / rough timeline: {horizon}

    Cluster constraints:
    - 1–2 questions are broad anchor questions (Role=CORE), describing the main uncertainty about this theme.
    - The remaining questions (Role=VARIANT) explore different angles: regions, actors, risk tails, adoption speed, distributional effects, policy scenarios, etc.

    Output ONLY the N blocks in the EXACT format specified in the system message.
    Do NOT restate the instructions. Do NOT explain your choices.
    """
)

# ---------------------- Judge for proto ----------------------

JUDGE_SYS_PROTO = """
You rate proto forecasting questions for Metaculus.

Your output is consumed by an automated parser which expects a SINGLE LINE.
If you output multiple lines or deviate from the format, the result is discarded.

You MUST output exactly ONE line with this format:
clarity=X; resolvability=Y; forecastability=Z; decision_relevance=U; cost_safety=V; verdict=ACCEPT|SOFT_REJECT|HARD_REJECT; rationale=TEXT

Where:
- X, Y, Z, U, V are integers from 1 (very bad) to 5 (excellent).
- verdict is one of: ACCEPT, SOFT_REJECT, HARD_REJECT (uppercase).
- TEXT is a short explanation (<= 250 characters) and MUST NOT contain semicolons.

Criteria:
- clarity: Is the question understandable and well-posed?
- resolvability: Does it look resolvable from public sources in principle?
- forecastability: Is the outcome non-trivial and not already determined?
- decision_relevance: Would the outcome matter for at least some decisions?
- cost_safety: Effort to operationalize/resolve and topic safety.

Do NOT rewrite the question. Do NOT add extra lines. No markdown, no JSON.
""".strip()

JUDGE_USER_TMPL_PROTO = textwrap.dedent(
    """
    Rate this proto forecasting question for Metaculus.

    Cluster seed (theme):
    {seed}

    Proto-question:
    Role: {role}
    Angle: {angle}
    Title: {title}
    Question: {question}
    Candidate-source: {source}

    You are only scoring this proto-question, not rewriting it.

    Return exactly one line:
    clarity=X; resolvability=Y; forecastability=Z; decision_relevance=U; cost_safety=V; verdict=ACCEPT|SOFT_REJECT|HARD_REJECT; rationale=TEXT
    """
)

# ---------------------- SIMPLE AGENT (verdict + p_auto_resolve + rationale) ----------------------

AGENT_SYS_SIMPLE = """
You are a resolution-focused publication agent for Metaculus proto-questions.

You are used in a LOOP: the system calls you once per proto-question.
EACH CALL is about ONE SINGLE proto-question. Other questions will be handled by other calls.

Your job for this ONE proto-question:
- Think briefly about how it could be resolved from public data (what sources, what measurable outcome).
- Estimate P-auto-resolve: probability (0.0–1.0) that an automated system with web access could resolve it at the intended time.
- Decide one of three publication verdicts:
  - ACCEPT (publishable as-is or with minor editorial tweaks),
  - SOFT_REJECT (interesting but has a fixable problem),
  - HARD_REJECT (should not be published in this form).
- Provide:
  - a very short “resolution_hint” sentence (<= 200 characters) that names the main sources / metric for resolution,
  - a very short “rationale” sentence (<= 250 characters) explaining the verdict.

FORMAT (STRICT)
Your output is parsed by a machine. You MUST output EXACTLY ONE LINE:

p_auto_resolve=X; verdict=ACCEPT|SOFT_REJECT|HARD_REJECT; resolution_hint=TEXT; rationale=TEXT

Where:
- X is a float between 0.0 and 1.0 (you may write 0.65, 0.7, 0.3, etc.).
- TEXT fields MUST NOT contain semicolons.
- No markdown, no extra spaces at the beginning of the line, no commentary before or after.

If you do anything else, your answer will be discarded by the pipeline.
""".strip()

AGENT_USER_TMPL_SIMPLE = textwrap.dedent(
    """
    You are evaluating a SINGLE proto forecasting question which has already passed a first quality screen.

    Seed (cluster theme, for context only):
    {seed}

    Proto-question (the ONLY one you must evaluate in this call):
    Role: {role}
    Angle: {angle}
    Title: {title}
    Question: {question}
    Candidate-source (family): {source_hint}

    First-pass judge scores (for context only, not to restate):
    clarity={clarity}, resolvability={resolvability}, forecastability={forecastability},
    decision_relevance={decision_relevance}, cost_safety={cost_safety}, judge_verdict={judge_verdict}

    Your output must be EXACTLY ONE LINE with this format:
    p_auto_resolve=X; verdict=ACCEPT|SOFT_REJECT|HARD_REJECT; resolution_hint=TEXT; rationale=TEXT

    Where:
    - X is in [0.0, 1.0].
    - resolution_hint is one concise sentence mentioning the main public sources and type of measurable outcome.
    - rationale is one concise sentence explaining why you chose this verdict.
    - Neither TEXT may contain semicolons.

    Do NOT mention other questions or the rest of the cluster.
    Do NOT add any lines before or after this one line.
    """
)


# ============================================================
# 4. PARSING HELPERS
# ============================================================

def parse_proto_questions_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Parse QUESTION i / Role / Title / Question / Angle / Candidate-source blocks.
    """
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    questions: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    def push_current():
        nonlocal current
        if not current:
            return
        if current.get("title") and current.get("question"):
            current.setdefault("role", "VARIANT")
            current.setdefault("angle", "")
            current.setdefault("candidate_source", "")
            questions.append(current)
        current = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.upper().startswith("QUESTION "):
            push_current()
            current = {
                "role": "VARIANT",
                "angle": "",
                "title": "",
                "question": "",
                "candidate_source": "",
            }
            continue
        if current is None:
            continue
        lower = line.lower()
        if lower.startswith("role:"):
            val = line.split(":", 1)[1].strip().upper()
            if val not in {"CORE", "VARIANT"}:
                val = "VARIANT"
            current["role"] = val
        elif lower.startswith("title:"):
            current["title"] = line.split(":", 1)[1].strip()
        elif lower.startswith("question:"):
            current["question"] = line.split(":", 1)[1].strip()
        elif lower.startswith("angle:"):
            current["angle"] = line.split(":", 1)[1].strip()
        elif lower.startswith("candidate-source:"):
            current["candidate_source"] = line.split(":", 1)[1].strip()

    push_current()
    return questions


def parse_judge_line_proto(line: str) -> Dict[str, Any]:
    line = line.strip()
    parts = [p.strip() for p in line.split(";") if p.strip()]
    mapping: Dict[str, str] = {}
    for seg in parts:
        if "=" not in seg:
            continue
        k, v = seg.split("=", 1)
        mapping[k.strip().lower()] = v.strip()

    def to_int(name: str, default: int = 0) -> int:
        try:
            return int(mapping.get(name, default))
        except Exception:
            return default

    clarity = to_int("clarity")
    resolvability = to_int("resolvability")
    forecastability = to_int("forecastability")
    decision_relevance = to_int("decision_relevance")
    cost_safety = to_int("cost_safety")

    verdict = mapping.get("verdict", "SOFT_REJECT").strip().upper()
    if verdict not in {"ACCEPT", "SOFT_REJECT", "HARD_REJECT"}:
        verdict = "SOFT_REJECT"

    rationale = mapping.get("rationale", "").replace(";", ",")

    scores = [
        clarity,
        resolvability,
        forecastability,
        decision_relevance,
        cost_safety,
    ]
    valid_scores = [s for s in scores if isinstance(s, (int, float)) and s > 0]
    if valid_scores:
        overall = round(sum(valid_scores) / len(valid_scores), 2)
    else:
        overall = 0.0

    return {
        "clarity": clarity,
        "resolvability": resolvability,
        "forecastability": forecastability,
        "decision_relevance": decision_relevance,
        "cost_safety": cost_safety,
        "verdict": verdict,
        "rationale": rationale,
        "overall": overall,
    }


def _extract_first_float(value: str, default: float = 0.0) -> float:
    if value is None:
        return default
    value = value.strip()
    try:
        return float(value)
    except Exception:
        pass
    m = re.search(r"[-+]?\d*\.?\d+", value)
    if not m:
        return default
    try:
        return float(m.group(0))
    except Exception:
        return default


def parse_agent_line_simple(line: str) -> Dict[str, Any]:
    """
    Parse agent simple line:
    p_auto_resolve=X; verdict=...; resolution_hint=TEXT; rationale=TEXT
    """
    line = line.strip()
    parts = [p.strip() for p in line.split(";") if p.strip()]
    mapping: Dict[str, str] = {}
    for seg in parts:
        if "=" not in seg:
            continue
        k, v = seg.split("=", 1)
        mapping[k.strip().lower()] = v.strip()

    p_val = _extract_first_float(mapping.get("p_auto_resolve", "0.0"), default=0.0)

    verdict = mapping.get("verdict", "").strip().upper()
    if verdict not in {"ACCEPT", "SOFT_REJECT", "HARD_REJECT"}:
        verdict = "SOFT_REJECT"

    resolution_hint = mapping.get("resolution_hint", "")
    rationale = mapping.get("rationale", "")

    return {
        "p_auto_resolve": p_val,
        "agent_verdict": verdict,
        "resolution_hint": resolution_hint.replace(";", ","),
        "agent_rationale": rationale.replace(";", ","),
    }


def compute_overall_score(
    clarity: int,
    resolvability: int,
    forecastability: int,
    decision_relevance: int,
    cost_safety: int,
) -> float:
    scores = [clarity, resolvability, forecastability, decision_relevance, cost_safety]
    valid = [s for s in scores if isinstance(s, (int, float)) and s > 0]
    if not valid:
        return 0.0
    return round(sum(valid) / len(valid), 2)


# ============================================================
# 5. PIPELINE – GENERATION / JUDGE / SIMPLE AGENT
# ============================================================

def mock_proto_questions(seed: str, n: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    prefix = seed.strip().split("\n")[0][:60] or "Example topic"
    for i in range(n):
        role = "CORE" if i == 0 else "VARIANT"
        angle = "anchor question" if i == 0 else f"variant angle #{i}"
        out.append(
            {
                "role": role,
                "angle": angle,
                "title": f"[MOCK] {prefix} – Q{i+1}",
                "question": "Will the mocked event occur by 2030?",
                "candidate_source": "Mock dataset / World Bank / Reuters",
            }
        )
    return out


def generate_proto_questions(
    seed: str,
    tags: List[str],
    horizon: str,
    n: int,
    model: str,
    dry_run: bool = False,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    if dry_run:
        questions = mock_proto_questions(seed, n)
        return {
            "questions": questions,
            "raw_output": "",
            "n_parsed": len(questions),
            "n_requested": n,
            "attempts": 1,
        }

    all_questions: List[Dict[str, Any]] = []
    raw_chunks: List[str] = []
    attempts = 0

    while len(all_questions) < n and attempts < max_attempts:
        attempts += 1
        need = n - len(all_questions)
        user_prompt = GEN_USER_TMPL_PROTO.format(
            n=need,
            seed=seed.strip(),
            tags=", ".join(tags) or "unspecified",
            horizon=horizon.strip() or "unspecified",
        )

        raw = call_openrouter_raw(
            messages=[
                {"role": "system", "content": GEN_SYS_PROTO},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            max_tokens=3500,
            temperature=0.5,
        )
        raw_chunks.append(raw)

        new_questions = parse_proto_questions_from_text(raw)
        if new_questions:
            all_questions.extend(new_questions)

    if not all_questions:
        raise RuntimeError("Generator returned no parsable proto-questions.")

    questions = all_questions[:n]
    raw_output = "\n\n-----\n\n".join(raw_chunks)

    return {
        "questions": questions,
        "raw_output": raw_output,
        "n_parsed": len(questions),
        "n_requested": n,
        "attempts": attempts,
    }


def judge_proto_question(
    q: Dict[str, Any],
    seed: str,
    model: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if dry_run:
        clarity = random.randint(3, 5)
        resolvability = random.randint(2, 5)
        forecastability = random.randint(2, 5)
        decision_relevance = random.randint(2, 5)
        cost_safety = random.randint(2, 5)
        overall = compute_overall_score(
            clarity, resolvability, forecastability, decision_relevance, cost_safety
        )
        verdict = random.choice(["ACCEPT", "SOFT_REJECT", "HARD_REJECT"])
        rationale = "Mock scores for dry-run."
        return {
            "clarity": clarity,
            "resolvability": resolvability,
            "forecastability": forecastability,
            "decision_relevance": decision_relevance,
            "cost_safety": cost_safety,
            "verdict": verdict,
            "rationale": rationale,
            "overall": overall,
        }

    user_text = JUDGE_USER_TMPL_PROTO.format(
        seed=seed.strip(),
        role=q.get("role", "VARIANT"),
        angle=q.get("angle", ""),
        title=q.get("title", ""),
        question=q.get("question", ""),
        source=q.get("candidate_source", ""),
    )
    raw = call_openrouter_raw(
        messages=[
            {"role": "system", "content": JUDGE_SYS_PROTO},
            {"role": "user", "content": user_text},
        ],
        model=model,
        max_tokens=512,
        temperature=0.0,
    )
    first_line = ""
    for ln in raw.splitlines():
        if ln.strip():
            first_line = ln.strip()
            break
    if not first_line:
        raise RuntimeError(f"Empty judge response: {raw!r}")

    return parse_judge_line_proto(first_line)


def judge_all_proto(
    questions: List[Dict[str, Any]],
    seed: str,
    model: str,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    scores: List[Dict[str, Any]] = []
    for q in questions:
        s = judge_proto_question(q, seed=seed, model=model, dry_run=dry_run)
        scores.append(s)
    return scores


def agent_eval_for_question_simple(
    q: Dict[str, Any],
    judge: Dict[str, Any],
    seed: str,
    model: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if dry_run:
        p = round(random.uniform(0.2, 0.9), 2)
        verdict = random.choice(["ACCEPT", "SOFT_REJECT", "HARD_REJECT"])
        return {
            "p_auto_resolve": p,
            "agent_verdict": verdict,
            "resolution_hint": "Mock resolution via generic public data.",
            "agent_rationale": "Mock agent rationale (dry-run).",
        }

    user_text = AGENT_USER_TMPL_SIMPLE.format(
        seed=seed.strip(),
        role=q.get("role", "VARIANT"),
        angle=q.get("angle", ""),
        title=q.get("title", ""),
        question=q.get("question", ""),
        source_hint=q.get("candidate_source", ""),
        clarity=judge.get("clarity", 0),
        resolvability=judge.get("resolvability", 0),
        forecastability=judge.get("forecastability", 0),
        decision_relevance=judge.get("decision_relevance", 0),
        cost_safety=judge.get("cost_safety", 0),
        judge_verdict=judge.get("verdict", ""),
    )

    raw = call_openrouter_raw(
        messages=[
            {"role": "system", "content": AGENT_SYS_SIMPLE},
            {"role": "user", "content": user_text},
        ],
        model=model,
        max_tokens=600,
        temperature=0.1,
    )

    first_line = ""
    for ln in raw.splitlines():
        if ln.strip():
            first_line = ln.strip()
            break
    if not first_line:
        return {
            "p_auto_resolve": 0.0,
            "agent_verdict": "SOFT_REJECT",
            "resolution_hint": "",
            "agent_rationale": "Empty agent response.",
        }

    return parse_agent_line_simple(first_line)


def agent_eval_top_k(
    questions: List[Dict[str, Any]],
    judge_scores: List[Dict[str, Any]],
    seed: str,
    model: str,
    dry_run: bool,
    top_k: int,
) -> List[Optional[Dict[str, Any]]]:
    """
    Judge ranks all questions and selects the K best.
    The agent processes ALL selected questions (K of them), one call per question.
    Questions outside the top-K are marked as DROPPED_BY_JUDGE.
    """
    n = len(questions)
    if n == 0:
        return []

    overall_list: List[float] = []
    for s in judge_scores:
        o = s.get("overall", 0.0)
        if not o or o <= 0:
            o = compute_overall_score(
                s.get("clarity", 0),
                s.get("resolvability", 0),
                s.get("forecastability", 0),
                s.get("decision_relevance", 0),
                s.get("cost_safety", 0),
            )
        overall_list.append(o)

    idx_sorted = sorted(range(n), key=lambda i: overall_list[i], reverse=True)
    agent_results: List[Optional[Dict[str, Any]]] = [None] * n

    k_eff = min(max(top_k, 0), n)
    for rank, idx in enumerate(idx_sorted):
        if rank >= k_eff:
            break
        agent_results[idx] = agent_eval_for_question_simple(
            questions[idx],
            judge_scores[idx],
            seed=seed,
            model=model,
            dry_run=dry_run,
        )

    return agent_results


# ============================================================
# 6. STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Metaculus – Proto Question Cluster + Judge + Simple Agent",
    page_icon="📊",
    layout="wide",
)

st.title("Metaculus – Proto Question Cluster + Judge + Simple Agent")

st.markdown(
    """
Pipeline:

1. Generate **X proto-questions** from a seed prompt, as a coherent cluster around one theme.
2. A **judge** scores each question on 5 criteria and ranks them.
3. The **K best questions** (by judge overall score) are sent to a **simple agent** that:
   - estimates **P-auto-resolve** (0–1),
   - outputs a short **resolution_hint** (how it would be resolved),
   - assigns a final verdict: `ACCEPT`, `SOFT_REJECT`, or `HARD_REJECT`,
   - gives a brief rationale for that verdict.

The agent’s output is a single, semicolon-separated line to keep parsing robust.
JSON only appears at download time.
"""
)

# ---------------------- Sidebar ----------------------

with st.sidebar:
    st.header("OpenRouter configuration")

    dry_run = st.checkbox(
        "Dry run (no API calls, mock questions & scores)",
        value=False,
    )

    api_key_input = st.text_input(
        "OpenRouter API key",
        type="password",
        help="Key will be kept only in this session.",
    )
    if api_key_input:
        st.session_state["OPENROUTER_API_KEY_OVERRIDE"] = api_key_input.strip()

    model_override = st.text_input(
        "OpenRouter model ID",
        value=OPENROUTER_MODEL_ENV or DEFAULT_MODEL,
        help="Example: openai/gpt-4o-mini, openai/gpt-5.1, anthropic/claude-3.5-sonnet, ...",
    )

    n = st.slider(
        "Number X of proto-questions to generate",
        min_value=1,
        max_value=30,
        value=10,
        step=1,
    )

    top_k = st.slider(
        "K best questions for agent (X ≥ K)",
        min_value=1,
        max_value=30,
        value=5,
        step=1,
    )

current_key = get_openrouter_key()
if not current_key and not dry_run:
    st.warning(
        "No OPENROUTER_API_KEY detected. Enter it in the sidebar, "
        "or enable dry_run mode for local testing."
    )

# ---------------------- Main inputs ----------------------

st.subheader("Seed prompt")

seed = st.text_area(
    "Seed prompt (1–12 sentences)",
    height=180,
    placeholder=(
        "Describe the central theme and uncertainty.\n"
        "Example: 'I want questions about the adoption of LLMs in universities worldwide, "
        "including distributional impacts, policy responses, and long-run productivity effects up to 2035.'"
    ),
)

tags_str = st.text_input(
    "Domain tags (comma-separated)",
    value="ai,policy,education",
)

horizon = st.text_input(
    "Horizon / rough timeline",
    value="resolve by 2035-12-31 UTC",
)

run_button = st.button("Run full pipeline (X → judge → K → agent)")


# ---------------------- Run pipeline ----------------------

if run_button:
    if not seed.strip():
        st.warning("Please provide a seed prompt.")
    elif not dry_run and not current_key:
        st.error("No OPENROUTER_API_KEY set and dry_run is disabled.")
    else:
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        model = (model_override or "").strip() or DEFAULT_MODEL

        if top_k > n:
            st.warning("K cannot be larger than X; K has been truncated to X.")
            top_k = n

        st.info(f"Using model: `{model}`")

        # 1) Generation
        with st.spinner("Generating proto-question cluster..."):
            try:
                gen_res = generate_proto_questions(
                    seed=seed,
                    tags=tags,
                    horizon=horizon,
                    n=n,
                    model=model,
                    dry_run=dry_run,
                )
            except Exception as e:
                st.error(f"Generation error: {e}")
                gen_res = None

        if gen_res is not None:
            questions = gen_res["questions"]
            n_parsed = gen_res.get("n_parsed", len(questions))
            n_requested = gen_res.get("n_requested", n)
            attempts = gen_res.get("attempts", 1)
            raw_output = gen_res.get("raw_output", "")

            if n_parsed < n_requested:
                st.warning(
                    f"Generator returned only {n_parsed} proto-questions out of requested {n_requested} "
                    f"after {attempts} attempt(s)."
                )

            if not questions:
                st.error("No proto-questions were parsed. Check generator prompts.")
            else:
                # 2) Judge
                with st.spinner("Judging proto-questions (first pass)..."):
                    try:
                        scores_list = judge_all_proto(
                            questions=questions,
                            seed=seed,
                            model=model,
                            dry_run=dry_run,
                        )
                    except Exception as e:
                        st.error(f"Judge error: {e}")
                        scores_list = None

                if scores_list is not None:
                    # 3) Agent on top-K
                    with st.spinner("Agent evaluating K best questions (P-auto-resolve + verdict)..."):
                        try:
                            agent_list = agent_eval_top_k(
                                questions=questions,
                                judge_scores=scores_list,
                                seed=seed,
                                model=model,
                                dry_run=dry_run,
                                top_k=top_k,
                            )
                        except Exception as e:
                            st.error(f"Agent error: {e}")
                            agent_list = [None] * len(questions)

                    # 4) Combine
                    entries: List[Dict[str, Any]] = []

                    overall_list = [
                        s.get("overall")
                        if s.get("overall")
                        else compute_overall_score(
                            s.get("clarity", 0),
                            s.get("resolvability", 0),
                            s.get("forecastability", 0),
                            s.get("decision_relevance", 0),
                            s.get("cost_safety", 0),
                        )
                        for s in scores_list
                    ]
                    idx_sorted = sorted(
                        range(len(questions)),
                        key=lambda i: overall_list[i],
                        reverse=True,
                    )
                    judge_rank_map = {idx: rank + 1 for rank, idx in enumerate(idx_sorted)}

                    for idx, q in enumerate(questions):
                        j = scores_list[idx]
                        a = agent_list[idx]

                        overall = j.get("overall")
                        if not overall or overall <= 0:
                            overall = compute_overall_score(
                                j.get("clarity", 0),
                                j.get("resolvability", 0),
                                j.get("forecastability", 0),
                                j.get("decision_relevance", 0),
                                j.get("cost_safety", 0),
                            )

                        if a is None:
                            final_verdict = "DROPPED_BY_JUDGE"
                            p_auto = 0.0
                        else:
                            final_verdict = a.get("agent_verdict", "SOFT_REJECT")
                            p_auto = a.get("p_auto_resolve", 0.0)

                        entries.append(
                            {
                                "question": q,
                                "judge": {
                                    **j,
                                    "overall": overall,
                                    "judge_rank": judge_rank_map[idx],
                                },
                                "agent": a,
                                "final": {
                                    "final_verdict": final_verdict,
                                    "p_auto_resolve": p_auto,
                                },
                            }
                        )

                    def severity_order(v: str) -> int:
                        v = (v or "").upper()
                        if v == "ACCEPT":
                            return 0
                        if v == "SOFT_REJECT":
                            return 1
                        if v == "HARD_REJECT":
                            return 2
                        if v == "DROPPED_BY_JUDGE":
                            return 3
                        return 2

                    entries_sorted = sorted(
                        entries,
                        key=lambda e: (
                            severity_order(e["final"]["final_verdict"]),
                            -e["judge"]["overall"],
                            -(e["final"]["p_auto_resolve"] or 0.0),
                        ),
                    )

                    res = {
                        "model": model,
                        "seed": seed,
                        "tags": tags,
                        "horizon": horizon,
                        "n_requested": n_requested,
                        "n_parsed": n_parsed,
                        "top_k": min(top_k, len(questions)),
                        "entries": entries_sorted,
                        "raw_output": raw_output,
                    }
                else:
                    res = None

                st.session_state["qgen_result"] = res

# ---------------------- Display results ----------------------

res = st.session_state.get("qgen_result")

if res is not None:
    model = res["model"]
    seed = res["seed"]
    tags = res["tags"]
    horizon = res["horizon"]
    entries = res["entries"]
    raw_output = res.get("raw_output", "")

    st.subheader("Last run summary")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"**Model:** `{model}`")
        st.markdown(
            f"**X generated (parsed):** {res['n_parsed']} "
            f"(requested {res['n_requested']})"
        )
    with col_b:
        st.markdown(f"**K sent to agent (top by judge):** {res['top_k']}")
        st.markdown(f"**Tags:** {', '.join(tags) if tags else '(none)'}")
    with col_c:
        st.markdown(f"**Horizon:** {horizon}")
        st.markdown("**Seed preview:**")
        st.caption(seed[:200] + ("..." if len(seed) > 200 else ""))

    st.info(
        "P-auto-resolve is the agent’s estimate (0–1) of the probability that an automated "
        "resolver could settle the question from public sources using the hinted resolution path."
    )

    # Flatten entries
    rows = []
    for idx, e in enumerate(entries, start=1):
        q = e["question"]
        j = e["judge"]
        a = e["agent"] or {}
        f = e["final"]

        rows.append(
            {
                "rank_overall": idx,
                "judge_rank": j.get("judge_rank", None),
                "final_verdict": f.get("final_verdict", ""),
                "p_auto_resolve": f.get("p_auto_resolve", 0.0),
                "overall_judge": j.get("overall", 0.0),
                "role": q.get("role", ""),
                "angle": q.get("angle", ""),
                "title": q.get("title", ""),
                "question": q.get("question", ""),
                "candidate_source": q.get("candidate_source", ""),
                "clarity": j.get("clarity"),
                "resolvability": j.get("resolvability"),
                "forecastability": j.get("forecastability"),
                "decision_relevance": j.get("decision_relevance"),
                "cost_safety": j.get("cost_safety"),
                "judge_verdict": j.get("verdict", ""),
                "judge_rationale": j.get("rationale", ""),
                "agent_verdict": a.get("agent_verdict", ""),
                "agent_resolution_hint": a.get("resolution_hint", ""),
                "agent_rationale": a.get("agent_rationale", ""),
            }
        )

    df = pd.DataFrame(rows)

    st.subheader("Evaluated proto-question cluster")

    filter_choice = st.selectbox(
        "Which questions to show?",
        [
            "Agent ACCEPT only",
            "Agent ACCEPT + SOFT_REJECT",
            "All questions (including HARD_REJECT and dropped)",
        ],
        index=0,
    )

    def keep_row(row) -> bool:
        v = (row["final_verdict"] or "").upper()
        if filter_choice == "Agent ACCEPT only":
            return v == "ACCEPT"
        elif filter_choice == "Agent ACCEPT + SOFT_REJECT":
            return v in ("ACCEPT", "SOFT_REJECT")
        else:
            return True

    df_filtered = df[df.apply(keep_row, axis=1)].copy()

    if not df_filtered.empty:
        top = df_filtered.iloc[0]

        st.markdown("### Best candidate (by final verdict, judge score, and P-auto-resolve)")

        st.markdown(f"**Role:** {top['role']} – **Angle:** {top['angle']}")
        st.markdown(f"**Title:** {top['title']}")
        st.markdown(f"**Proto-question:** {top['question']}")
        st.markdown(f"**Candidate-source (family):** {top['candidate_source']}")

        st.markdown(
            f"**Final verdict (agent):** {top['final_verdict']} – "
            f"P-auto-resolve={top['p_auto_resolve']:.2f}, "
            f"Judge overall={top['overall_judge']:.2f} (rank={top['judge_rank']})"
        )
        st.markdown(
            f"**Judge scores:** clarity={top['clarity']}, resolvability={top['resolvability']}, "
            f"forecastability={top['forecastability']}, decision_relevance={top['decision_relevance']}, "
            f"cost_safety={top['cost_safety']} (verdict={top['judge_verdict']})"
        )
        if isinstance(top.get("judge_rationale"), str) and top["judge_rationale"]:
            st.markdown(f"**Judge rationale:** {top['judge_rationale']}")

        if isinstance(top.get("agent_verdict"), str) and top["agent_verdict"]:
            st.markdown("**Agent decision & resolution hint:**")
            st.markdown(
                f"- **Agent verdict:** {top['agent_verdict']} "
                f"(P-auto-resolve={top['p_auto_resolve']:.2f})"
            )
            if top.get("agent_resolution_hint"):
                st.markdown(f"- **Resolution hint:** {top['agent_resolution_hint']}")
            if top.get("agent_rationale"):
                st.markdown(f"- **Agent rationale:** {top['agent_rationale']}")

        st.markdown("### Table of proto-questions")
        st.caption(
            "Sorted by final verdict (agent), judge overall score, and P-auto-resolve. "
            "Questions beyond the K best are marked as DROPPED_BY_JUDGE (no agent evaluation)."
        )
        st.dataframe(
            df_filtered[
                [
                    "rank_overall",
                    "judge_rank",
                    "final_verdict",
                    "p_auto_resolve",
                    "overall_judge",
                    "role",
                    "angle",
                    "title",
                    "question",
                    "agent_verdict",
                    "agent_resolution_hint",
                    "agent_rationale",
                ]
            ],
            use_container_width=True,
        )
    else:
        st.info("No rows match the selected filter.")

    # Downloads
    st.subheader("Download")

    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    export_list = []
    for e in entries:
        export_list.append(
            {
                "seed": seed,
                "question": e["question"],
                "judge": e["judge"],
                "agent": e["agent"],
                "final": e["final"],
            }
        )

    json_bytes = _json.dumps(
        {
            "model": model,
            "seed": seed,
            "tags": tags,
            "horizon": horizon,
            "entries": export_list,
        },
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    col1, col2 = st.columns(2)
    col1.download_button(
        "Download CSV (flat view)",
        data=csv_bytes,
        file_name="metaculus_proto_cluster_pipeline.csv",
        mime="text/csv",
    )
    col2.download_button(
        "Download JSON (nested)",
        data=json_bytes,
        file_name="metaculus_proto_cluster_pipeline.json",
        mime="application/json",
    )

    with st.expander("Raw generation output (debug)"):
        if raw_output:
            st.code(raw_output, language="text")
        else:
            st.caption("No raw generator output stored (dry_run or mock mode).")
else:
    st.info("Configure X, K and the seed prompt, then click the button to run the full pipeline.")
