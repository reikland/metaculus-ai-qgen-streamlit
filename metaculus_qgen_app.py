#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import io
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
TITLE = "Metaculus – Question Generator + Judge"

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
        "User-Agent": ascii_safe("metaculus-qgen-judge/0.1"),
    }


def call_openrouter_raw(
    messages: List[Dict[str, str]],
    model: str,
    max_tokens: int = 2000,
    temperature: float = 0.4,
    retries: int = 3,
) -> str:
    """Appel brut à OpenRouter, retourne simplement le texte de réponse."""
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
# 3. JSON PARSING – SIMPLE & ROBUST
# ============================================================

def extract_code_fence(s: str) -> Optional[str]:
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def balanced_slice(s: str, open_char: str, close_char: str) -> Optional[str]:
    start = s.find(open_char)
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == open_char:
            depth += 1
        elif c == close_char:
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None


def parse_json_relaxed(s: str) -> Any:
    """
    Essaie fort de récupérer un JSON valide depuis la sortie du modèle.

    On :
    - enlève les ``` et "json",
    - supprime les virgules finales avant } ou ],
    - tente de parser la chaîne entière,
    - sinon, on essaie de trouver un bloc [ ... ] ou { ... } équilibré.
    """
    s = s.strip()

    # Enlève les fences éventuelles
    s = re.sub(r"```(?:json)?", "", s, flags=re.IGNORECASE)
    s = s.replace("```", "")

    # Supprime les virgules finales avant } ou ]
    s = re.sub(r",(\s*[}\]])", r"\1", s)

    # 1) tentative directe
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) bloc array
    blk = balanced_slice(s, "[", "]")
    if blk:
        try:
            return json.loads(blk)
        except Exception:
            pass

    # 3) bloc objet
    blk = balanced_slice(s, "{", "}")
    if blk:
        try:
            return json.loads(blk)
        except Exception:
            pass

    # 4) chaque { ... } individuellement
    objs = []
    for m in re.finditer(r"\{.*?\}", s, flags=re.DOTALL):
        try:
            objs.append(json.loads(m.group(0)))
        except Exception:
            continue
    if objs:
        return objs if len(objs) > 1 else objs[0]

    raise ValueError("Could not parse JSON from model output")


# ============================================================
# 4. PROMPTS – GÉNÉRATION + JUGE (PLAUDIBILITÉ)
# ============================================================

GEN_SYS = """
You are a JSON API, not a chat assistant.

Your ONLY task is to output machine-readable JSON that can be parsed by json.loads in Python
and JSON.parse in JavaScript without any preprocessing.

Hard constraints:
- Output EXACTLY ONE JSON value, which MUST be a JSON array of objects.
- NO text before or after the array.
- NO markdown, NO code fences, NO comments, NO explanations.
- Use only double quotes for strings. Never use single quotes.
- Never include trailing commas.
- Allowed literals: null, true, false, numbers, strings, arrays, objects.
- Never output NaN, Infinity, or -Infinity.

If you violate these constraints, the calling application will fail.
Return STRICT JSON only.
""".strip()

GEN_USER_TMPL = textwrap.dedent(
    """
    Task: Generate {n} Metaculus-style forecasting questions.

    Topic brief (3–6 lines):
    {brief}

    Domain tags: {tags}
    Target horizon (if relevant): {horizon}

    For EACH question, output an object with EXACTLY these keys:
      - "title": short, <= 100 characters
      - "body": 2–5 sentences describing context
      - "resolution_criteria": exact steps to decide YES/NO or measure the outcome
      - "timeframe": an object with "start", "end", "timezone" (ISO-like strings, timezone = "UTC")
      - "answer_type": one of "binary", "numeric", "date", "multiple"

    Substantive constraints:
    - Outcomes must be resolvable from public sources (official statistics, reputable newswires, etc.).
    - Include explicit end dates (UTC) and clear thresholds.
    - Questions should be PLAUSIBLE: numeric thresholds, dates and quantities must be in realistic
      ranges consistent with current public information and known orders of magnitude.
      Avoid arbitrary or random-looking numbers (e.g. "17,345,678,901") when you have no basis.
      If you are uncertain, prefer:
        - ranges ("between 10% and 30%"),
        - relative comparisons ("higher than in 2024"),
        - or coarse thresholds (e.g. "above 3× the 2024 value"),
      instead of a very specific figure that is likely to be wrong.
    - If you have internet/tools, you MAY look up up-to-date facts, but still output only JSON.
    - Avoid questions already resolved at time of writing; there must be genuine uncertainty.

    OUTPUT FORMAT (CRITICAL):
    - Return a SINGLE JSON ARRAY of EXACTLY {n} objects.
    - Do NOT wrap it in markdown.
    - Do NOT add any explanation or commentary.
    - Do NOT include trailing commas.

    Example of the STRUCTURE only (values are placeholders, but this is valid JSON):
    [
      {{
        "title": "Example question title",
        "body": "Example body text.",
        "resolution_criteria": "How the question will be resolved.",
        "timeframe": {{
          "start": "2025-01-01 00:00:00",
          "end": "2030-12-31 23:59:59",
          "timezone": "UTC"
        }},
        "answer_type": "binary"
      }}
    ]
    """
)

JUDGE_SYS = """
You are a JSON API, not a chat assistant.

You score candidate forecasting questions.

Your ONLY task is to output machine-readable JSON that can be parsed by json.loads in Python
and JSON.parse in JavaScript without any preprocessing.

Constraints:
- Output EXACTLY ONE JSON object.
- NO markdown, NO code fences, NO comments, NO explanations.
- Use only double quotes for strings. Never use single quotes.
- Never include trailing commas, NaN, Infinity, or -Infinity.

Return STRICT JSON only.
""".strip()

JUDGE_USER_TMPL = textwrap.dedent(
    """
    Evaluate the following forecasting question for Metaculus-style use.

    You MUST return a SINGLE JSON object with EXACTLY these keys:
      - "clarity": integer from 1 to 5
      - "operationalization": integer from 1 to 5 (how well it can be mechanically resolved)
      - "plausibility": integer from 1 to 5 (are the numbers, dates, and scenarios realistic and
        consistent with known orders of magnitude, not obviously pulled from thin air?)
      - "usefulness": integer from 1 to 5 (decision/forecasting value)
      - "safety": integer from 1 to 5 (no obvious policy/abuse/harassment issues)
      - "overall": float from 1 to 5 (mean of the 5 scores, rounded to 2 decimals)
      - "notes": a short justification string (<= 300 characters)

    Example of valid JSON output (values are illustrative only):
    {{
      "clarity": 5,
      "operationalization": 4,
      "plausibility": 4,
      "usefulness": 5,
      "safety": 5,
      "overall": 4.60,
      "notes": "Clear question; uses realistic ranges and resolvable statistics."
    }}

    Remember:
    - Do NOT output markdown or code fences.
    - Do NOT include comments or trailing commas.

    Candidate question (JSON):
    {candidate_json}
    """
)


# ============================================================
# 5. CORE GENERATION LOGIC
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
                "resolution_criteria": (
                    "On 2030-12-31 23:59:59 UTC, check source X; "
                    "resolve YES if condition Y holds, otherwise NO."
                ),
                "timeframe": {
                    "start": "2025-01-01 00:00:00",
                    "end": "2030-12-31 23:59:59",
                    "timezone": "UTC",
                },
                "answer_type": "binary",
            }
        )
    return out


def generate_questions(
    brief: str,
    tags: List[str],
    horizon: str,
    n: int,
    model: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Génère des questions, sans étape intermédiaire.
    Retourne un dict avec :
      - "model"
      - "questions" (liste d'objets normalisés)
      - "raw_output" (texte brut du modèle, pour debug)
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

    try:
        data = parse_json_relaxed(raw)
    except Exception as e:
        snippet = raw[:600].replace("\n", "\\n")
        raise RuntimeError(
            f"Model output was not valid JSON: {e}. "
            f"First 600 chars: {snippet!r}"
        ) from e

    # ======================================================
    # Normalisation de la forme du JSON -> liste de questions
    # ======================================================
    questions: List[Dict[str, Any]] = []

    if isinstance(data, list):
        # Cas idéal : le modèle renvoie déjà un tableau
        questions = data

    elif isinstance(data, dict):
        # 1) Clé "questions"
        if isinstance(data.get("questions"), list):
            questions = data["questions"]
        else:
            # 2) Chercher une valeur qui soit une liste de dicts
            for v in data.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    questions = v
                    break

            # 3) Sinon, interpréter le dict lui-même comme UNE question
            if not questions:
                questions = [data]
    else:
        raise RuntimeError(
            f"Parsed JSON is neither an array nor an object; got type {type(data).__name__}."
        )

    # ==========================================
    # Normalisation champ par champ des questions
    # ==========================================
    norm_questions: List[Dict[str, Any]] = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        tf = q.get("timeframe") or {}
        if not isinstance(tf, dict):
            tf = {}
        norm_questions.append(
            {
                "title": q.get("title", ""),
                "body": q.get("body", ""),
                "resolution_criteria": q.get("resolution_criteria", ""),
                "timeframe": {
                    "start": tf.get("start", ""),
                    "end": tf.get("end", ""),
                    "timezone": tf.get("timezone", ""),
                },
                "answer_type": q.get("answer_type", ""),
            }
        )

    if not norm_questions:
        raise RuntimeError("No valid question objects found in JSON.")

    return {"model": model, "questions": norm_questions, "raw_output": raw}


# ============================================================
# 6. JUDGE LOGIC (SCORE + TOP K)
# ============================================================

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
        operationalization = random.randint(3, 5)
        plausibility = random.randint(3, 5)
        usefulness = random.randint(3, 5)
        safety = 5
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
            "overall": overall,
            "notes": "Mock scores for dry-run mode.",
        }

    candidate_json = json.dumps(question, ensure_ascii=False)
    user = JUDGE_USER_TMPL.format(candidate_json=candidate_json)

    raw = call_openrouter_raw(
        messages=[
            {"role": "system", "content": JUDGE_SYS},
            {"role": "user", "content": user},
        ],
        model=model,
        max_tokens=800,
        temperature=0.0,
    )

    try:
        data = parse_json_relaxed(raw)
    except Exception as e:
        snippet = raw[:400].replace("\n", "\\n")
        raise RuntimeError(
            f"Judge output was not valid JSON: {e}. "
            f"First 400 chars: {snippet!r}"
        ) from e

    if not isinstance(data, dict):
        raise RuntimeError("Judge JSON is not an object.")

    def as_int(name: str, default: int = 3) -> int:
        v = data.get(name, default)
        try:
            return int(round(float(v)))
        except Exception:
            return default

    clarity = as_int("clarity")
    operationalization = as_int("operationalization")
    plausibility = as_int("plausibility")
    usefulness = as_int("usefulness")
    safety = as_int("safety", default=5)

    try:
        overall = float(data.get("overall", 0.0))
        if overall <= 0:
            raise ValueError()
    except Exception:
        overall = round(
            (clarity + operationalization + plausibility + usefulness + safety) / 5.0,
            2,
        )

    notes = str(data.get("notes", "") or "")

    return {
        "clarity": clarity,
        "operationalization": operationalization,
        "plausibility": plausibility,
        "usefulness": usefulness,
        "safety": safety,
        "overall": round(overall, 2),
        "notes": notes,
    }


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
# 7. STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Metaculus – Question Generator + Judge",
    page_icon="📊",
    layout="wide",
)

st.title("Metaculus – Question Generator + Judge")

st.markdown(
    """
Simple pipeline:
1. Generate N Metaculus-style questions (JSON only).
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
# 8. RUN PIPELINE ON BUTTON CLICK
# ------------------------------------------------------------

if run_button:
    if not brief.strip():
        st.warning("Please provide at least a short topic brief.")
    elif not dry_run and not current_key:
        st.error("No OPENROUTER_API_KEY set and dry_run is disabled.")
    else:
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        model = (model_override or "").strip() or DEFAULT_MODEL

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
# 9. DISPLAY LAST RESULT (PERSISTENT)
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
        tf = q.get("timeframe") or {}
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
                "resolution_criteria": q.get("resolution_criteria", ""),
                "timeframe_start": tf.get("start", ""),
                "timeframe_end": tf.get("end", ""),
                "timezone": tf.get("timezone", ""),
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
    st.markdown(f"**Resolution criteria:** {top['resolution_criteria']}")
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
    json_bytes = json.dumps(
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
        st.code(raw_output, language="json")
else:
    st.info("Configure your topic and click 'Generate + Judge questions' to start.")

