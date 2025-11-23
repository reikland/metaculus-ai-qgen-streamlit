#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import re
import time
import textwrap
from typing import Dict, Any, List, Optional

import requests
import pandas as pd
import streamlit as st

# ============================================================
# 1. CONFIG
# ============================================================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL_ENV = os.environ.get("OPENROUTER_MODEL", "").strip()

REFERER = "https://localhost"
TITLE = "Metaculus – Question Generator + Judge (Text Template)"

DEFAULT_MODEL = "openai/gpt-4o-mini"

# Stocke le dernier résultat pour éviter les resets
if "qgen_result" not in st.session_state:
    st.session_state["qgen_result"] = None


# ============================================================
# 2. OPENROUTER HELPERS
# ============================================================

def get_openrouter_key() -> str:
    """Récupère la clé OpenRouter depuis la sidebar, l'env ou Streamlit secrets."""
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
    """Encode en ASCII safe pour les headers HTTP."""
    try:
        return s.encode("latin-1", "ignore").decode("latin-1")
    except Exception:
        return "".join(ch for ch in s if ord(ch) < 256)


def or_headers() -> Dict[str, str]:
    """Headers standards OpenRouter."""
    key = get_openrouter_key()
    if not key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Referer": ascii_safe(REFERER),
        "X-Title": ascii_safe(TITLE),
        "User-Agent": ascii_safe("metaculus-qgen-judge-text/0.1"),
    }


def call_openrouter_raw(
    messages: List[Dict[str, str]],
    model: str,
    max_tokens: int = 2000,
    temperature: float = 0.4,
    retries: int = 3,
) -> str:
    """Appel brut à OpenRouter, retourne simplement le texte de réponse."""
    import json as _json

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
# 3. PROMPTS – GÉNÉRATION + JUGE (TEMPLATE TEXTE)
# ============================================================

GEN_SYS = """
You generate forecasting questions in a STRICT, line-based text template.

Rules (CRITICAL):
- Do NOT use markdown, bullets, JSON, or code fences.
- Do NOT add any commentary before the first question or after the last one.
- All fields must be on ONE line (no internal line breaks).
- You MUST respect the template exactly, with the same field labels and order.
""".strip()

GEN_USER_TMPL = textwrap.dedent(
    """
    Task: Generate {n} Metaculus-style forecasting questions (true questions i.e with a "?" at the end of the sentence).

    Topic brief (3–6 lines):
    {brief}

    Domain tags: {tags}
    Target horizon (if relevant): {horizon}

    For EACH question i = 1..{n}, you MUST output a block with EXACTLY this template:

    QUESTION i
    Title: <short title, <= 100 characters>
    Body: <2–5 sentences, BUT all on a single line (no line breaks)>
    Resolution: <exact steps to resolve (who/what/when/where), single line>
    Timeframe-start: <YYYY-MM-DD HH:MM:SS UTC or empty if unknown>
    Timeframe-end: <YYYY-MM-DD HH:MM:SS UTC (target resolution time), or empty>
    Timezone: UTC
    Answer-type: <one of: binary, numeric, date, multiple>

    Constraints on content:
    - Outcomes must be resolvable from public sources (official statistics, reputable newswires, etc.).
    - Include explicit end dates (UTC) when possible and clear thresholds.
    - Questions should be PLAUSIBLE: numeric thresholds, dates and quantities must be in realistic
      ranges consistent with current public information and known orders of magnitude.
      Avoid arbitrary or random-looking numbers (e.g. "17,345,678,901") when you have no basis.
      If you are uncertain, prefer:
        - ranges ("between 10% and 30%"),
        - relative comparisons ("higher than in 2024"),
        - or coarse thresholds (e.g. "above 3× the 2024 value"),
      instead of a very specific figure that is likely to be wrong.
    - Avoid questions already resolved at time of writing; there must be genuine uncertainty.

    OUTPUT FORMAT (CRITICAL):
    - You MUST output EXACTLY {n} question blocks, numbered QUESTION 1, QUESTION 2, ..., QUESTION {n}.
    - Each block MUST follow the template above, in the same order and with the same labels.
    - Separate blocks with a single blank line.
    - Do NOT add any other lines, explanations, headings or comments.
    """
)

JUDGE_SYS = """
You rate forecasting questions in a single semicolon-separated line.

Rules (CRITICAL):
- Output exactly ONE line of text.
- Format must be:
  clarity=X; operationalization=Y; plausibility=Z; usefulness=U; safety=S; overall=O; notes=TEXT
- X,Y,Z,U,S are integers 1–5.
- O is a float 1–5 (mean of the five scores, rounded to 2 decimals).
- Do NOT use semicolons in notes; use commas instead.
- No markdown, no extra lines, no JSON.
""".strip()

JUDGE_USER_TMPL = textwrap.dedent(
    """
    Rate the following forecasting question for Metaculus-style use on 5 dimensions:
    - clarity (1–5),
    - operationalization (1–5) – how well it can be mechanically resolved,
    - plausibility (1–5) – are numbers/dates/scenarios realistic vs known orders of magnitude,
    - usefulness (1–5) – decision/forecast value,
    - safety (1–5) – policy/abuse issues.

    Then compute overall as the mean of the 5 scores, rounded to 2 decimals.

    You MUST respond with EXACTLY ONE line:
    clarity=X; operationalization=Y; plausibility=Z; usefulness=U; safety=S; overall=O; notes=TEXT

    Where:
    - X,Y,Z,U,S are integers 1–5,
    - O is a float 1–5,
    - TEXT is a short justification (<= 300 characters),
    - Do NOT use semicolons in TEXT.

    Question:
    Title: {title}
    Body: {body}
    Resolution: {resolution}
    Timeframe-start: {tstart}
    Timeframe-end: {tend}
    Timezone: {tz}
    Answer-type: {atype}
    """
)


# ============================================================
# 4. PARSING – QUESTIONS & SCORES
# ============================================================

def mock_questions(brief: str, n: int) -> List[Dict[str, Any]]:
    """Mode démo sans API : fabrique des questions factices."""
    out: List[Dict[str, Any]] = []
    prefix = brief.strip().split("\n")[0][:60] or "Example topic"
    for i in range(n):
        out.append(
            {
                "title": f"[MOCK] {prefix} – Q{i+1}",
                "body": "This is a mock question body for testing the UI.",
                "resolution": (
                    "On 2030-12-31 23:59:59 UTC, check source X; "
                    "resolve YES if condition Y holds, otherwise NO."
                ),
                "timeframe_start": "2025-01-01 00:00:00",
                "timeframe_end": "2030-12-31 23:59:59",
                "timezone": "UTC",
                "answer_type": "binary",
            }
        )
    return out


def parse_questions_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Parse la sortie du générateur au format :

    QUESTION i
    Title: ...
    Body: ...
    Resolution: ...
    Timeframe-start: ...
    Timeframe-end: ...
    Timezone: ...
    Answer-type: ...
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    questions: List[Dict[str, Any]] = []

    current: Optional[Dict[str, Any]] = None

    def push_current():
        nonlocal current
        if current and (current.get("title") or current.get("body")):
            questions.append(current)
        current = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        m_q = re.match(r"^QUESTION\s+(\d+)", line, flags=re.IGNORECASE)
        if m_q:
            # Nouvelle question
            push_current()
            current = {
                "title": "",
                "body": "",
                "resolution": "",
                "timeframe_start": "",
                "timeframe_end": "",
                "timezone": "",
                "answer_type": "",
            }
            continue

        if current is None:
            # Ignore tout texte avant la première "QUESTION"
            continue

        # Parsing champ: "Key: value"
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip().lower()
        val = val.strip()

        if key == "title":
            current["title"] = val
        elif key == "body":
            current["body"] = val
        elif key == "resolution":
            current["resolution"] = val
        elif key == "timeframe-start":
            current["timeframe_start"] = val
        elif key == "timeframe-end":
            current["timeframe_end"] = val
        elif key == "timezone":
            current["timezone"] = val
        elif key == "answer-type":
            current["answer_type"] = val

    # Dernière question
    push_current()
    return questions


def parse_judge_line(line: str) -> Dict[str, Any]:
    """
    Parse une ligne du type :
    clarity=X; operationalization=Y; plausibility=Z; usefulness=U; safety=S; overall=O; notes=TEXT
    """
    line = line.strip()
    # On sépare sur ';' mais en gardant tout ce qui dépasse pour notes
    parts = [p.strip() for p in line.split(";")]
    # On veut au moins 6 segments pour les scores, le reste = notes
    if len(parts) < 6:
        raise ValueError(f"Judge line too short: {line!r}")

    mapping: Dict[str, Any] = {}
    def parse_val(segment: str) -> (str, str):
        if "=" not in segment:
            return segment.strip().lower(), ""
        k, v = segment.split("=", 1)
        return k.strip().lower(), v.strip()

    # clarity, operationalization, plausibility, usefulness, safety, overall
    keys_expected = ["clarity", "operationalization", "plausibility",
                     "usefulness", "safety", "overall"]

    for i, key_name in enumerate(keys_expected):
        if i >= len(parts):
            break
        k, v = parse_val(parts[i])
        if k != key_name:
            # clé inattendue, on essaie quand même d'interpréter
            k = key_name
        mapping[k] = v

    # notes = le reste
    if len(parts) > len(keys_expected):
        notes_raw = ";".join(parts[len(keys_expected):]).strip()
        # enlever éventuel "notes=" au début
        if notes_raw.lower().startswith("notes="):
            notes_raw = notes_raw[6:].strip()
        mapping["notes"] = notes_raw
    else:
        mapping["notes"] = ""

    # Convertir les scores
    def to_int(name: str, default: int = 3) -> int:
        try:
            return int(round(float(mapping.get(name, default))))
        except Exception:
            return default

    def to_float(name: str, default: float = 0.0) -> float:
        try:
            return float(mapping.get(name, default))
        except Exception:
            return default

    clarity = to_int("clarity")
    operationalization = to_int("operationalization")
    plausibility = to_int("plausibility")
    usefulness = to_int("usefulness")
    safety = to_int("safety", default=5)
    overall = to_float("overall", default=0.0)

    if overall <= 0:
        overall = round(
            (clarity + operationalization + plausibility + usefulness + safety) / 5.0,
            2,
        )

    return {
        "clarity": clarity,
        "operationalization": operationalization,
        "plausibility": plausibility,
        "usefulness": usefulness,
        "safety": safety,
        "overall": round(overall, 2),
        "notes": mapping.get("notes", ""),
    }


# ============================================================
# 5. PIPELINE – GÉNÉRATION & JUGE
# ============================================================

def generate_questions(
    brief: str,
    tags: List[str],
    horizon: str,
    n: int,
    model: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Génère des questions au format texte structuré, puis parse en liste de dicts.
    """
    if dry_run:
        qs = mock_questions(brief, n)
        return {"model": "[MOCK]", "questions": qs, "raw_output": "(mock)"}

    user = GEN_USER_TMPL.format(
        n=n,
        brief=brief,
        tags=",".join(tags),
        horizon=horizon,
    )

    raw = call_openrouter_raw(
        messages=[
            {"role": "system", "content": GEN_SYS},
            {"role": "user", "content": user},
        ],
        model=model,
        max_tokens=4000,
        temperature=0.5,
    )

    questions = parse_questions_from_text(raw)
    if not questions:
        raise RuntimeError(
            "Could not parse any questions from model output. "
            "Check format and model obedience."
        )

    return {"model": model, "questions": questions, "raw_output": raw}


def judge_question(
    question: Dict[str, Any],
    model: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Retourne un dict avec les clés :
      "clarity", "operationalization", "plausibility",
      "usefulness", "safety", "overall", "notes".
    """
    if dry_run:
        import random
        clarity = random.randint(3, 5)
        oper = random.randint(3, 5)
        plaus = random.randint(3, 5)
        use = random.randint(3, 5)
        safety = 5
        overall = round((clarity + oper + plaus + use + safety) / 5.0, 2)
        return {
            "clarity": clarity,
            "operationalization": oper,
            "plausibility": plaus,
            "usefulness": use,
            "safety": safety,
            "overall": overall,
            "notes": "Mock scores for dry-run mode.",
        }

    user = JUDGE_USER_TMPL.format(
        title=question.get("title", ""),
        body=question.get("body", ""),
        resolution=question.get("resolution", ""),
        tstart=question.get("timeframe_start", ""),
        tend=question.get("timeframe_end", ""),
        tz=question.get("timezone", ""),
        atype=question.get("answer_type", ""),
    )

    raw = call_openrouter_raw(
        messages=[
            {"role": "system", "content": JUDGE_SYS},
            {"role": "user", "content": user},
        ],
        model=model,
        max_tokens=400,
        temperature=0.0,
    )

    try:
        return parse_judge_line(raw)
    except Exception as e:
        raise RuntimeError(f"Failed to parse judge output: {e}. Raw: {raw!r}") from e


def judge_all_questions(
    questions: List[Dict[str, Any]],
    model: str,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    scores: List[Dict[str, Any]] = []
    for q in questions:
        s = judge_question(q, model=model, dry_run=dry_run)
        scores.append(s)
    return scores


# ============================================================
# 6. STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Metaculus – Question Generator + Judge",
    page_icon="📊",
    layout="wide",
)

st.title("Metaculus – Question Generator + Judge (Text Template)")

st.markdown(
    """
Pipeline:
1. Generate N Metaculus-style questions with a strict line-based template (no JSON).
2. Score each question on clarity, operationalization, PLAUSIBILITY, usefulness, safety.
3. Keep the top K questions by overall score.
"""
)

with st.sidebar:
    st.header("Settings")

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
        help=(
            "Examples: openai/gpt-4o-mini, openai/gpt-5.1, "
            "anthropic/claude-3.5-sonnet, qwen/qwen-2.5-32b-instruct..."
        ),
    )

    n = st.slider(
        "Number of questions to generate (N)",
        min_value=1,
        max_value=20,
        value=8,
        step=1,
    )

    top_k = st.slider(
        "Top K questions to keep",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    dry_run = st.checkbox(
        "Dry run (no API calls, mock questions & scores)",
        value=False,
    )

current_key = get_openrouter_key()
if not current_key and not dry_run:
    st.warning(
        "No OPENROUTER_API_KEY detected. Enter it in the sidebar, "
        "or set it as an environment variable / Streamlit secret."
    )

st.subheader("Problem setup")

brief = st.text_area(
    "Topic brief (3–6 lines)",
    height=150,
    placeholder="Describe the domain, actors, uncertainty and plausible ranges you care about...",
)

tags_str = st.text_input(
    "Domain tags (comma-separated)",
    value="ai,policy,geopolitics",
)

horizon = st.text_input(
    "Horizon / resolution description",
    value="resolve by 2035-12-31 UTC",
)

run_button = st.button("Generate + Judge questions")

# ------------------------------------------------------------
# 7. RUN PIPELINE ON BUTTON CLICK
# ------------------------------------------------------------

if run_button:
    if not brief.strip():
        st.warning("Please provide at least a short topic brief.")
    elif not dry_run and not current_key:
        st.error("No OPENROUTER_API_KEY set and dry_run is disabled.")
    else:
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        model = (model_override or "").strip() or DEFAULT_MODEL

        st.info(f"Using model: `{model}`")

        with st.spinner("Generating questions..."):
            try:
                gen_res = generate_questions(
                    brief=brief,
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

            with st.spinner("Judging questions..."):
                try:
                    scores_list = judge_all_questions(
                        questions=questions,
                        model=model,
                        dry_run=dry_run,
                    )
                except Exception as e:
                    st.error(f"Judge error: {e}")
                    scores_list = None

            if scores_list is not None:
                # Combine questions + scores, puis tri par overall
                entries = []
                for q, s in zip(questions, scores_list):
                    entries.append({"question": q, "scores": s})

                entries_sorted = sorted(
                    entries,
                    key=lambda e: e["scores"]["overall"],
                    reverse=True,
                )
                effective_k = min(len(entries_sorted), top_k)
                entries_sorted = entries_sorted[:effective_k]

                res = {
                    "model": model,
                    "entries": entries_sorted,
                    "raw_output": gen_res["raw_output"],
                }
            else:
                res = None
        else:
            res = None

        st.session_state["qgen_result"] = res

# ------------------------------------------------------------
# 8. DISPLAY LAST RESULT (PERSISTENT)
# ------------------------------------------------------------

res = st.session_state.get("qgen_result")

if res is not None:
    model = res["model"]
    entries = res["entries"]
    raw_output = res["raw_output"]

    st.subheader("Model used")
    st.write(model)

    # Tableau structuré pour affichage / export
    rows = []
    for e in entries:
        q = e["question"]
        s = e["scores"]
        rows.append(
            {
                "overall": s.get("overall", 0.0),
                "clarity": s.get("clarity"),
                "operationalization": s.get("operationalization"),
                "plausibility": s.get("plausibility"),
                "usefulness": s.get("usefulness"),
                "safety": s.get("safety"),
                "title": q.get("title", ""),
                "body": q.get("body", ""),
                "resolution": q.get("resolution", ""),
                "timeframe_start": q.get("timeframe_start", ""),
                "timeframe_end": q.get("timeframe_end", ""),
                "timezone": q.get("timezone", ""),
                "answer_type": q.get("answer_type", ""),
                "judge_notes": s.get("notes", ""),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("overall", ascending=False)

    st.subheader("Top-K judged questions")
    st.dataframe(df, use_container_width=True, height=500)

    top = df.iloc[0]
    st.markdown("### Best question (rank #1)")
    st.markdown(f"**Title:** {top['title']}")
    st.markdown(f"**Body:** {top['body']}")
    st.markdown(f"**Resolution:** {top['resolution']}")
    st.markdown(
        f"**Timeframe:** {top['timeframe_start']} → "
        f"{top['timeframe_end']} ({top['timezone'] or 'UTC'})"
    )
    st.markdown(f"**Answer type:** {top['answer_type']}")
    st.markdown(
        f"**Scores:** overall={top['overall']} "
        f"(clarity={top['clarity']}, operationalization={top['operationalization']}, "
        f"plausibility={top['plausibility']}, usefulness={top['usefulness']}, "
        f"safety={top['safety']})"
    )
    if top.get("judge_notes"):
        st.markdown(f"**Judge notes:** {top['judge_notes']}")

    # Downloads
    st.subheader("Download")

    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    export_list = []
    for e in entries:
        export_list.append(
            {
                "question": e["question"],
                "scores": e["scores"],
            }
        )
    import json as _json
    json_bytes = _json.dumps(
        {"model": model, "entries": export_list},
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    col1, col2 = st.columns(2)
    col1.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="metaculus_topK_questions.csv",
        mime="text/csv",
    )
    col2.download_button(
        "Download JSON",
        data=json_bytes,
        file_name="metaculus_topK_questions.json",
        mime="application/json",
    )

    # Zone optionnelle pour inspecter la sortie brute du générateur
    with st.expander("Raw generation output (debug)"):
        st.code(raw_output, language="text")
else:
    st.info("Configure your topic and click 'Generate + Judge questions' to start.")
