#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import time
import textwrap
import random
from typing import Dict, Any, List, Optional

import requests
import pandas as pd
import streamlit as st
import json as _json

# ============================================================
# 1. CONFIG / CONSTANTS
# ============================================================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL_ENV = os.environ.get("OPENROUTER_MODEL", "").strip()

REFERER = "https://localhost"
TITLE = "Metaculus – Evolutionary Proto Question Generator"

# Modèle principal (génération + extension)
DEFAULT_MAIN_MODEL = OPENROUTER_MODEL_ENV or "openai/gpt-5.1"
# Modèle léger (judge résolvabilité + info)
DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"

if "evo_result" not in st.session_state:
    st.session_state["evo_result"] = None


# ============================================================
# 2. OPENROUTER HELPERS
# ============================================================

def get_openrouter_key() -> str:
    """Récupère la clé OpenRouter de plusieurs sources possibles."""
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
        "User-Agent": ascii_safe("metaculus-evo-qgen/0.1"),
    }


def call_openrouter_raw(
    messages: List[Dict[str, str]],
    model: str,
    max_tokens: int = 2000,
    temperature: float = 0.4,
    retries: int = 3,
) -> str:
    """Appel brut à OpenRouter, en forçant un format court et strict côté modèle."""
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
# 3. PROMPTS – GÉNÉRATEUR / JUDGE / EXPANDER
# ============================================================

# ---------------------- GÉNÉRATION INITIALE (N questions) ----------------------

GEN_SYS_INITIAL = """
You generate CLUSTERS of proto forecasting questions for Metaculus.

Your output is parsed by a STRICT machine.
If you deviate from the required format, THE OUTPUT IS DISCARDED.

ABSOLUTE RULES
- You MUST strictly follow the template described below.
- You MUST produce EXACTLY N questions: not fewer, not more.
- You MUST NOT add explanations, comments, headings, or markdown.
- You MUST NOT include internal reasoning, rationale, or examples.
- Your VERY FIRST non-empty line MUST be: "QUESTION 1".
- Your LAST non-empty line MUST start with "Candidate-source:" for QUESTION N.
- No JSON. No code fences. No bullet lists. Plain text only.

CLUSTER BEHAVIOUR
- Interpret the seed as describing ONE central theme (cluster theme).
- Produce a coherent cluster of N related proto-questions.
- 1–2 questions should be broad "anchor" questions about the central theme (Role=CORE).
- The remaining questions should be narrower "variants" exploring different angles (Role=VARIANT).

Content constraints:
- Questions must be about an uncertain future or an as-yet-unobserved outcome.
- They must be resolvable from public data (official statistics, major datasets, government reports, reputable newswires).
- Avoid trivial questions whose probability is obviously ~0% or ~100%.
- Avoid questions that are already resolved.

STRICT FORMAT (LINE-BASED)
For each i = 1..N you output a block with these 5 lines:

QUESTION i
Role: CORE or VARIANT
Title: <short title, <= 100 characters, single line>
Question: <1–3 sentences, single line, ends with '?' or equivalent>
Angle: <short phrase capturing the angle within the cluster>
Candidate-source: <likely family of public resolution sources/datasets, single line>

Between blocks you MAY optionally have a single blank line.
You MUST NOT output anything else before, between, or after the blocks.
""".strip()

GEN_USER_TMPL_INITIAL = textwrap.dedent(
    """
    You must now generate a CLUSTER of proto forecasting questions.

    HARD CONSTRAINT:
    - N_questions = {n}.
    - You MUST output EXACTLY N_questions blocks, labelled QUESTION 1, QUESTION 2, ..., QUESTION {n}.
    - Any extra text or missing block makes the output INVALID.

    Seed (central theme):
    {seed}

    Optional context:
    - Domain tags: {tags}
    - Horizon / rough timeline: {horizon}

    Cluster constraints:
    - 1–2 questions are broad anchor questions (Role=CORE).
    - Remaining questions (Role=VARIANT) explore distinct angles: geographies, actors, baselines vs tails, policy vs market, distributional effects, etc.

    Output ONLY the blocks in the EXACT format specified in the system message.
    Do NOT restate the instructions. Do NOT explain your choices.
    """
)

# ---------------------- JUDGE LIGHT (keep K parmi N) ----------------------

JUDGE_SYS_KEEP = """
You are a FAST, STRICT judge for proto forecasting questions.

Your ONLY task is to decide whether to KEEP or DISCARD ONE question, based on:
- resolvability from public sources,
- information value for forecasting and decision-making.

You MUST output EXACTLY ONE LINE, with this format:

keep=0|1; resolvability=X; info=Y; rationale=TEXT

Hard constraints:
- X and Y MUST be integers from 1 to 5.
- rationale MUST be <= 200 characters and MUST NOT contain semicolons.
- The very first non-space characters of your reply MUST be "keep=".
- You MUST NOT add any other lines, JSON, markdown, or commentary.
- No bullet lists. No explanations before or after the line.
- If you are unsure, choose a reasonable guess and still follow the format.

Scoring hints:
- resolvability: 1 = barely or not resolvable; 5 = clearly resolvable from stable public sources with clear resolution time.
- info: 1 = almost no useful information for real decisions; 5 = high value-of-information for policies, investment, planning, or safety.
""".strip()

JUDGE_USER_TMPL_KEEP = textwrap.dedent(
    """
    You are judging the following proto forecasting question.

    Seed / cluster theme:
    {seed}

    Horizon:
    {horizon}

    Domain tags (optional):
    {tags}

    Proto-question:
    Title: {title}
    Question: {question}
    Candidate-source: {source}

    Decision rule:
    - keep=1 if the question is clearly resolvable from public sources AND has at least moderate information value.
    - keep=0 otherwise.

    Now output ONLY the single line:
    keep=0|1; resolvability=X; info=Y; rationale=TEXT
    """
)

# ---------------------- EXTENSION / MUTATION (factor par question gardée) ----------------------

EXP_SYS = """
You generate MUTATED proto forecasting questions for Metaculus.

Your output is parsed by a STRICT machine.
If you deviate from the required format, THE OUTPUT IS DISCARDED.

ABSOLUTE RULES
- You MUST strictly follow the template described below.
- For each parent question, you MUST produce EXACTLY M children.
- You MUST NOT add explanations, comments, headings, or markdown.
- You MUST NOT include internal reasoning, rationale, or examples.
- Your VERY FIRST non-empty line MUST be: "QUESTION 1".
- The LAST non-empty line MUST start with "Candidate-source:" for QUESTION (M * number_of_parents).
- No JSON. No code fences. No bullet lists. Plain text only.

MUTATION BEHAVIOUR
- You receive a list of "parent" questions that are already decent.
- For each parent, you generate M new questions which are MUTATIONS of that parent:

Mutation rules:
- Keep the core uncertainty / theme similar.
- For each child, change AT LEAST ONE of:
  - geography or population,
  - metric or threshold,
  - NOT time horizon,
  - scenario (baseline vs tail),
  - policy vs market framing,
  - level of aggregation.
- Avoid trivial paraphrases; each child must be meaningfully different.
- Ensure every child is clearly resolvable from a family of public sources (stats, datasets, official reports, reputable news wires).
- Avoid obviously 0%/100% questions and already-resolved outcomes.

STRICT FORMAT (LINE-BASED)
You output a GLOBAL list of QUESTION blocks, with indices from 1..(M * number_of_parents).

Each block has 5 lines:

QUESTION i
Role: CORE or VARIANT
Title: <short title, <= 100 characters, single line>
Question: <1–3 sentences, single line, ends with '?' or equivalent>
Angle: <short phrase indicating how it mutates its parent>
Candidate-source: <likely family of public resolution sources/datasets, single line>

You MAY optionally insert a blank line between blocks.
You MUST NOT group by parent explicitly; the machine only relies on block indices.
You MUST NOT output anything else before, between, or after the blocks.
""".strip()

EXP_USER_TMPL = textwrap.dedent(
    """
    You will receive K parent proto forecasting questions.

    For EACH parent, you MUST generate EXACTLY M mutated child questions.
    - M = {m}
    - Total number of QUESTION blocks you output MUST be: total_blocks = K * M.
    - If K * M = {total_blocks}, you MUST output exactly {total_blocks} blocks.

    Parents (read-only, DO NOT restate them in your output):

    {parents_block}

    Remember:
    - Do NOT mention the parent IDs or numbers in your output.
    - Follow exactly the QUESTION block format described in the system message.
    - Output ONLY the QUESTION blocks, nothing else.
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


def parse_judge_keep_line(line: str) -> Dict[str, Any]:
    """
    Parse:
      keep=0|1; resolvability=X; info=Y; rationale=TEXT
    """
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

    keep = to_int("keep", 0)
    if keep not in (0, 1):
        keep = 0
    resolvability = to_int("resolvability", 0)
    info = to_int("info", 0)

    rationale = mapping.get("rationale", "").replace(";", ",").strip()
    if len(rationale) > 300:
        rationale = rationale[:300]

    return {
        "keep": keep,
        "resolvability": resolvability,
        "info": info,
        "rationale": rationale,
        "raw_line": line,
    }


# ============================================================
# 5. PIPELINE FUNCTIONS
# ============================================================

# ---------------------- MOCK HELPERS (dry_run) ----------------------

def mock_proto_questions(seed: str, n: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    prefix = seed.strip().split("\n")[0][:60] or "Example topic"
    for i in range(n):
        role = "CORE" if i < 2 else "VARIANT"
        angle = "anchor question" if i < 2 else f"variant angle #{i}"
        out.append(
            {
                "role": role,
                "angle": angle,
                "title": f"[MOCK] {prefix} – Q{i+1}",
                "question": "Will the mocked event occur before 2035-12-31?",
                "candidate_source": "Mock dataset / World Bank / Reuters",
            }
        )
    return out


# ---------------------- Étape 0 – Génération initiale ----------------------

def generate_initial_questions(
    seed: str,
    tags: List[str],
    horizon: str,
    n: int,
    model: str,
    dry_run: bool = False,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    """Génère N proto-questions (generation=0)."""
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

        user_prompt = GEN_USER_TMPL_INITIAL.format(
            n=need,
            seed=seed.strip(),
            tags=", ".join(tags) or "unspecified",
            horizon=horizon.strip() or "unspecified",
        )

        raw = call_openrouter_raw(
            messages=[
                {"role": "system", "content": GEN_SYS_INITIAL},
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


# ---------------------- Étape 1 – Judge (résolvabilité + info) ----------------------

def judge_one_question_keep(
    q: Dict[str, Any],
    seed: str,
    tags: List[str],
    horizon: str,
    judge_model: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if dry_run:
        keep = random.choice([0, 1])
        resolvability = random.randint(1, 5)
        info = random.randint(1, 5)
        rationale = "Mock judge (dry run)."
        return {
            "keep": keep,
            "resolvability": resolvability,
            "info": info,
            "rationale": rationale,
            "raw_line": f"keep={keep}; resolvability={resolvability}; info={info}; rationale={rationale}",
        }

    user_text = JUDGE_USER_TMPL_KEEP.format(
        seed=seed.strip(),
        horizon=horizon.strip(),
        tags=", ".join(tags) or "unspecified",
        title=q.get("title", ""),
        question=q.get("question", ""),
        source=q.get("candidate_source", ""),
    )

    raw = call_openrouter_raw(
        messages=[
            {"role": "system", "content": JUDGE_SYS_KEEP},
            {"role": "user", "content": user_text},
        ],
        model=judge_model,
        max_tokens=256,
        temperature=0.0,
    )

    first_line = ""
    for ln in raw.splitlines():
        if "keep=" in ln:
            first_line = ln.strip()
            break
    if not first_line:
        raise RuntimeError(f"Judge returned no parsable line: {raw!r}")

    parsed = parse_judge_keep_line(first_line)
    return parsed


def judge_initial_questions(
    questions: List[Dict[str, Any]],
    seed: str,
    tags: List[str],
    horizon: str,
    judge_model: str,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for q in questions:
        res = judge_one_question_keep(
            q=q,
            seed=seed,
            tags=tags,
            horizon=horizon,
            judge_model=judge_model,
            dry_run=dry_run,
        )
        results.append(res)
    return results


def select_top_k(
    questions: List[Dict[str, Any]],
    judge_res: List[Dict[str, Any]],
    k: int,
) -> Dict[str, Any]:
    """
    Sélectionne K questions:
    - priorité aux keep=1,
    - tri par resolvability puis info,
    - fill-up avec keep=0 si nécessaire.
    """
    n = len(questions)
    if n != len(judge_res):
        raise ValueError("questions and judge_res length mismatch")

    indices = list(range(n))

    def score_tuple(i: int):
        jr = judge_res[i]
        # higher resolvability, then info
        return (jr.get("resolvability", 0), jr.get("info", 0))

    kept_indices = [i for i in indices if judge_res[i].get("keep", 0) == 1]
    not_kept_indices = [i for i in indices if i not in kept_indices]

    # tri décroissant (resolvability, info)
    kept_indices_sorted = sorted(kept_indices, key=score_tuple, reverse=True)
    not_kept_sorted = sorted(not_kept_indices, key=score_tuple, reverse=True)

    final_indices: List[int] = []

    # d'abord les keep=1
    for i in kept_indices_sorted:
        if len(final_indices) >= k:
            break
        final_indices.append(i)

    # si pas assez, compléter avec les meilleurs "keep=0"
    if len(final_indices) < k:
        for i in not_kept_sorted:
            if len(final_indices) >= k:
                break
            final_indices.append(i)

    # Au final, on ne garde que min(k, n)
    final_indices = final_indices[: min(k, n)]

    # Marquage keep_final
    keep_final_flags = [False] * n
    for idx in final_indices:
        keep_final_flags[idx] = True

    return {
        "final_indices": final_indices,
        "keep_final_flags": keep_final_flags,
        "n_kept_initial": len(kept_indices),
        "n_selected_final": len(final_indices),
    }


# ---------------------- Étape 2 – Extension / mutation ----------------------

def build_parents_block(parents: List[Dict[str, Any]]) -> str:
    """
    Texte lisible pour le modèle d'extension. Ne sera PAS parsé, juste lu par le LLM.
    """
    chunks = []
    for i, p in enumerate(parents, start=1):
        chunk = textwrap.dedent(
            f"""
            PARENT {i}
            Role: {p.get('role', 'VARIANT')}
            Title: {p.get('title', '')}
            Question: {p.get('question', '')}
            Angle: {p.get('angle', '')}
            Candidate-source: {p.get('candidate_source', '')}
            """
        ).strip()
        chunks.append(chunk)
    return "\n\n".join(chunks)


def expand_questions_around_kept(
    parents: List[Dict[str, Any]],
    seed: str,
    tags: List[str],
    horizon: str,
    m: int,
    model: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Génère m enfants par parent via le modèle principal.
    On fait ici UNE SEULE requête globale (tous les parents d'un coup).
    """
    if not parents or m <= 0:
        return {"children": [], "raw_output": ""}

    if dry_run:
        children: List[Dict[str, Any]] = []
        for p_idx, p in enumerate(parents):
            for j in range(m):
                children.append(
                    {
                        "role": "VARIANT",
                        "angle": f"mock child #{j+1} of parent {p_idx+1}",
                        "title": f"[MOCK CHILD] {p.get('title', '')} – var {j+1}",
                        "question": "Mock child question, mutated around the parent theme?",
                        "candidate_source": p.get("candidate_source", ""),
                    }
                )
        return {
            "children": children,
            "raw_output": "",
        }

    parents_block = build_parents_block(parents)
    total_blocks = len(parents) * m

    user_text = EXP_USER_TMPL.format(
        m=m,
        total_blocks=total_blocks,
        parents_block=parents_block,
    )

    raw = call_openrouter_raw(
        messages=[
            {"role": "system", "content": EXP_SYS},
            {"role": "user", "content": user_text},
        ],
        model=model,
        max_tokens=4000,
        temperature=0.6,
    )

    children = parse_proto_questions_from_text(raw)

    return {
        "children": children,
        "raw_output": raw,
    }


# ============================================================
# 6. STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Metaculus – Evolutionary Proto Question Generator",
    page_icon="🧬",
    layout="wide",
)

st.title("Metaculus – Evolutionary Proto Question Generator")

st.markdown(
    """
Pipeline en 3 étapes :

1. **Génération initiale** de N proto-questions (modèle principal).
2. **Judge light** qui évalue résolvabilité + information value et conserve K questions.
3. **Extension évolutive** : pour chaque question gardée, le modèle principal génère `factor` nouvelles questions (mutations).

Le formatage est strictement contrôlé (pas de JSON renvoyé par les modèles) et le JSON final est construit uniquement côté Python.
"""
)

# ---------------------- Sidebar ----------------------

with st.sidebar:
    st.header("Configuration OpenRouter")

    dry_run = st.checkbox(
        "Dry run (no API calls, mock questions & scores)",
        value=False,
    )

    api_key_input = st.text_input(
        "OpenRouter API key",
        type="password",
        help="Key is only stored in this session.",
    )
    if api_key_input:
        st.session_state["OPENROUTER_API_KEY_OVERRIDE"] = api_key_input.strip()

    main_model_input = st.text_input(
        "Main model (generation + expansion)",
        value=DEFAULT_MAIN_MODEL,
        help="Ex: openai/gpt-5.1, anthropic/claude-3.5-sonnet, etc.",
    )

    judge_model_input = st.text_input(
        "Judge model (light, keep K)",
        value=DEFAULT_JUDGE_MODEL,
        help="Ex: openai/gpt-4o-mini (rapide, peu coûteux).",
    )

    st.markdown("---")

    n_initial = st.slider(
        "N initial proto-questions",
        min_value=5,
        max_value=40,
        value=20,
        step=1,
    )

    k_keep = st.slider(
        "K questions kept by judge",
        min_value=1,
        max_value=n_initial,
        value=min(10, n_initial),
        step=1,
    )

    expansion_factor = st.slider(
        "Expansion factor (children per kept question)",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Nombre de nouvelles questions générées par question gardée.",
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
    "Seed prompt (1–12 sentences, central theme)",
    height=180,
    placeholder=(
        "Describe the main uncertainty / topic.\n"
        "Example: 'I want questions about global diffusion of frontier AI systems in education and public administration "
        "by 2040, including inequalities, safety, and regulation.'"
    ),
)

tags_str = st.text_input(
    "Domain tags (comma-separated)",
    value="ai,policy,macro",
)

horizon = st.text_input(
    "Horizon / rough timeline",
    value="resolve by 2040-12-31 UTC",
)

run_button = st.button("Run full evolutionary pipeline (N → K → K×factor)")

# ---------------------- Run pipeline ----------------------

if run_button:
    if not seed.strip():
        st.warning("Please provide a seed prompt.")
    elif not dry_run and not current_key:
        st.error("No OPENROUTER_API_KEY set and dry_run is disabled.")
    else:
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        main_model = (main_model_input or "").strip() or DEFAULT_MAIN_MODEL
        judge_model = (judge_model_input or "").strip() or DEFAULT_JUDGE_MODEL

        st.info(
            f"Using main model (generation + expansion): `{main_model}`\n\n"
            f"Using judge model (keep K): `{judge_model}`"
        )

        # 1) Génération initiale
        with st.spinner("Step 1/3 – Generating initial proto-question cluster..."):
            try:
                gen_res = generate_initial_questions(
                    seed=seed,
                    tags=tags,
                    horizon=horizon,
                    n=n_initial,
                    model=main_model,
                    dry_run=dry_run,
                )
            except Exception as e:
                st.error(f"Generation error: {e}")
                gen_res = None

        if gen_res is not None:
            questions0 = gen_res["questions"]
            raw_gen_output = gen_res.get("raw_output", "")
            n_parsed = gen_res.get("n_parsed", len(questions0))
            n_requested = gen_res.get("n_requested", n_initial)
            attempts = gen_res.get("attempts", 1)

            if n_parsed < n_requested:
                st.warning(
                    f"Generator returned only {n_parsed} proto-questions out of requested {n_requested} "
                    f"after {attempts} attempt(s)."
                )

            if not questions0:
                st.error("No proto-questions were parsed. Check generator prompts.")
            else:
                # 2) Judge (résolvabilité + info)
                with st.spinner("Step 2/3 – Judging initial questions (keep K)..."):
                    try:
                        judge_res0 = judge_initial_questions(
                            questions=questions0,
                            seed=seed,
                            tags=tags,
                            horizon=horizon,
                            judge_model=judge_model,
                            dry_run=dry_run,
                        )
                    except Exception as e:
                        st.error(f"Judge error: {e}")
                        judge_res0 = None

                if judge_res0 is not None:
                    selection = select_top_k(
                        questions=questions0,
                        judge_res=judge_res0,
                        k=k_keep,
                    )

                    final_indices = selection["final_indices"]
                    keep_final_flags = selection["keep_final_flags"]
                    n_kept_initial = selection["n_kept_initial"]
                    n_selected_final = selection["n_selected_final"]

                    st.info(
                        f"Judge initial keep=1 count: {n_kept_initial} / {n_initial}; "
                        f"Selected for expansion (final K): {n_selected_final} (target K={k_keep})."
                    )

                    # Attribuer IDs et génération
                    initial_entries: List[Dict[str, Any]] = []
                    id_map = {}  # index -> id

                    for idx, q in enumerate(questions0):
                        q_id = f"g0-q{idx+1}"
                        id_map[idx] = q_id
                        jr = judge_res0[idx]
                        initial_entries.append(
                            {
                                "id": q_id,
                                "generation": 0,
                                "parent_id": None,
                                "role": q.get("role", ""),
                                "angle": q.get("angle", ""),
                                "title": q.get("title", ""),
                                "question": q.get("question", ""),
                                "candidate_source": q.get("candidate_source", ""),
                                "judge_keep": jr.get("keep", 0),
                                "judge_resolvability": jr.get("resolvability", 0),
                                "judge_info": jr.get("info", 0),
                                "judge_rationale": jr.get("rationale", ""),
                                "judge_raw_line": jr.get("raw_line", ""),
                                "keep_final": bool(keep_final_flags[idx]),
                            }
                        )

                    parents_for_expansion: List[Dict[str, Any]] = []
                    parents_ids: List[str] = []
                    for idx in final_indices:
                        parents_for_expansion.append(questions0[idx])
                        parents_ids.append(id_map[idx])

                    # 3) Extension / mutation
                    with st.spinner("Step 3/3 – Expanding around kept questions (evolutionary step)..."):
                        try:
                            exp_res = expand_questions_around_kept(
                                parents=parents_for_expansion,
                                seed=seed,
                                tags=tags,
                                horizon=horizon,
                                m=expansion_factor,
                                model=main_model,
                                dry_run=dry_run,
                            )
                        except Exception as e:
                            st.error(f"Expansion error: {e}")
                            exp_res = None

                    expanded_entries: List[Dict[str, Any]] = []
                    raw_exp_output = ""
                    if exp_res is not None:
                        children = exp_res.get("children", [])
                        raw_exp_output = exp_res.get("raw_output", "")
                        # Attribution parent_id par ordre
                        # On suppose que le modèle respecte "M enfants par parent" dans l'ordre.
                        expected_children = len(parents_for_expansion) * expansion_factor
                        if len(children) != expected_children:
                            st.warning(
                                f"Expansion returned {len(children)} children, "
                                f"expected {expected_children} (K={len(parents_for_expansion)}, factor={expansion_factor})."
                            )
                        child_counter = 0
                        for p_local_idx, parent_id in enumerate(parents_ids):
                            for j in range(expansion_factor):
                                if child_counter >= len(children):
                                    break
                                c = children[child_counter]
                                c_id = f"g1-{p_local_idx+1}-{j+1}"
                                expanded_entries.append(
                                    {
                                        "id": c_id,
                                        "generation": 1,
                                        "parent_id": parent_id,
                                        "role": c.get("role", ""),
                                        "angle": c.get("angle", ""),
                                        "title": c.get("title", ""),
                                        "question": c.get("question", ""),
                                        "candidate_source": c.get("candidate_source", ""),
                                    }
                                )
                                child_counter += 1

                    # Construction du résultat global
                    evo_result = {
                        "models": {
                            "main": main_model,
                            "judge": judge_model,
                        },
                        "params": {
                            "n_initial": n_initial,
                            "k_keep": k_keep,
                            "expansion_factor": expansion_factor,
                        },
                        "seed": seed,
                        "tags": tags,
                        "horizon": horizon,
                        "initial": initial_entries,
                        "expanded": expanded_entries,
                        "raw_generation_output": raw_gen_output,
                        "raw_expansion_output": raw_exp_output,
                    }

                    st.session_state["evo_result"] = evo_result
                else:
                    st.session_state["evo_result"] = None
        else:
            st.session_state["evo_result"] = None

# ---------------------- Display results ----------------------

res = st.session_state.get("evo_result")

if res is not None:
    main_model = res["models"]["main"]
    judge_model = res["models"]["judge"]
    seed = res["seed"]
    tags = res["tags"]
    horizon = res["horizon"]
    initial_entries = res["initial"]
    expanded_entries = res["expanded"]

    st.subheader("Run summary")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Main model (generation + expansion):** `{main_model}`")
        st.markdown(f"**Judge model (keep K):** `{judge_model}`")
    with col2:
        st.markdown(f"**N initial proto-questions:** {res['params']['n_initial']}")
        st.markdown(f"**K target kept:** {res['params']['k_keep']}")
    with col3:
        st.markdown(f"**Expansion factor:** {res['params']['expansion_factor']}")
        st.markdown(f"**Horizon:** {horizon}")

    st.markdown("**Seed preview:**")
    st.caption(seed[:250] + ("..." if len(seed) > 250 else ""))

    # Table initiale
    st.subheader("Initial proto-questions (generation 0)")

    df_init = pd.DataFrame(initial_entries)

    if not df_init.empty:
        df_init_view = df_init[
            [
                "id",
                "keep_final",
                "judge_keep",
                "judge_resolvability",
                "judge_info",
                "title",
                "question",
                "candidate_source",
                "angle",
                "judge_rationale",
            ]
        ].copy()

        st.caption(
            "All generation-0 proto-questions with judge scores (resolvability, info) "
            "and final selection flag (keep_final)."
        )
        st.dataframe(df_init_view, use_container_width=True)
    else:
        st.info("No initial entries.")

    # Table expanded
    st.subheader("Expanded questions (generation 1)")

    df_exp = pd.DataFrame(expanded_entries)

    if not df_exp.empty:
        df_exp_view = df_exp[
            [
                "id",
                "parent_id",
                "role",
                "title",
                "question",
                "candidate_source",
                "angle",
            ]
        ].copy()

        st.caption(
            "Generation-1 questions (mutations), with parent_id referencing generation-0 questions."
        )
        st.dataframe(df_exp_view, use_container_width=True)
    else:
        st.info("No expanded entries (either expansion failed or factor=0).")

    # Debug / raw outputs
    with st.expander("Debug: raw model outputs"):
        st.markdown("**Raw generation output (initial cluster):**")
        raw_gen = res.get("raw_generation_output") or ""
        if raw_gen:
            st.code(raw_gen, language="text")
        else:
            st.caption("No stored raw generation output (dry_run or mock).")

        st.markdown("**Raw expansion output (mutations):**")
        raw_exp = res.get("raw_expansion_output") or ""
        if raw_exp:
            st.code(raw_exp, language="text")
        else:
            st.caption("No stored raw expansion output (dry_run or mock).")

        st.markdown("**Judge raw lines (keep=...; ...):**")
        if not df_init.empty:
            lines = df_init[["id", "judge_raw_line"]].to_dict(orient="records")
            for row in lines:
                st.code(f"{row['id']}: {row['judge_raw_line']}", language="text")
        else:
            st.caption("No judge lines available.")

    # Download JSON
    st.subheader("Download JSON")

    json_bytes = _json.dumps(
        res,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    st.download_button(
        "Download evolutionary run (JSON)",
        data=json_bytes,
        file_name="metaculus_evolutionary_proto_questions.json",
        mime="application/json",
    )
else:
    st.info("Configure N, K, factor and the seed, then click the button to run the evolutionary pipeline.")
