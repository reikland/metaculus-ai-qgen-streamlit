import os, time, csv, json, requests, itertools, random, re, io, tempfile, textwrap
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import streamlit as st

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS = "https://openrouter.ai/api/v1/models"
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "").strip()
REFERER = "https://localhost"
TITLE = "Metaculus AI Question Generation - Panel v1"
PREFERRED_MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1",
    "anthropic/claude-3.5-sonnet",
    "qwen/qwen-2.5-32b-instruct",
    "qwen/qwen-2.5-7b-instruct",
    "mistralai/mistral-large-2411",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-2-9b-it:free",
]

METACULUS_API2 = "https://www.metaculus.com/api2"
METACULUS_UA = {"User-Agent": "metaculus-question-scraper/0.1 (+python-requests)"}
METACULUS_HTTP = requests.Session()


def mc_get(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r = METACULUS_HTTP.get(url, params=params or {}, headers=METACULUS_UA, timeout=30)
    if r.status_code == 429:
        wait = float(r.headers.get("Retry-After", "1") or 1)
        time.sleep(min(wait, 10))
        r = METACULUS_HTTP.get(url, params=params or {}, headers=METACULUS_UA, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_metaculus_recent_questions(n_questions: int = 20, page_limit: int = 80) -> List[Dict[str, Any]]:
    data = mc_get(f"{METACULUS_API2}/questions/", {"status": "open", "limit": page_limit})
    results = data.get("results") or data.get("data") or []

    def ts(q: Dict[str, Any]) -> str:
        return (
            q.get("open_time")
            or q.get("created_time")
            or q.get("created_at")
            or q.get("scheduled_close_time")
            or ""
        )

    results.sort(key=ts, reverse=True)
    out: List[Dict[str, Any]] = []
    for q in results[:n_questions]:
        qid = q.get("id")
        if not qid:
            continue
        title = q.get("title") or f"Question {qid}"
        body = (
            q.get("description")
            or q.get("body")
            or q.get("background")
            or q.get("text")
            or ""
        )
        resolution_criteria = (
            q.get("resolution_criteria")
            or q.get("resolution")
            or q.get("resolution_text")
            or ""
        )
        open_time = q.get("open_time") or ""
        close_time = q.get("close_time") or q.get("scheduled_close_time") or ""
        timeframe = f"{open_time} -> {close_time}".strip(" ->")
        answer_type = (
            q.get("possibility_type")
            or q.get("possibility_space")
            or q.get("type")
            or ""
        )
        url = (
            q.get("page_url")
            or q.get("url")
            or f"https://www.metaculus.com/questions/{qid}/"
        )
        out.append(
            {
                "id": qid,
                "url": url,
                "title": title,
                "body": body,
                "resolution_criteria": resolution_criteria,
                "timeframe": timeframe,
                "answer_type": answer_type,
            }
        )
    return out


def scrape_metaculus_examples_to_csv(n: int, out_path: Optional[str] = None) -> str:
    n = max(10, min(n, 50))
    if out_path is None:
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        out_path = path
    qs = fetch_metaculus_recent_questions(n_questions=n, page_limit=max(80, n))
    fieldnames = [
        "id",
        "url",
        "title",
        "body",
        "resolution_criteria",
        "timeframe",
        "answer_type",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for q in qs:
            w.writerow(
                {
                    "id": q.get("id", ""),
                    "url": q.get("url", ""),
                    "title": q.get("title", ""),
                    "body": q.get("body", ""),
                    "resolution_criteria": q.get("resolution_criteria", ""),
                    "timeframe": q.get("timeframe", ""),
                    "answer_type": q.get("answer_type", ""),
                }
            )
    return out_path


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
        "User-Agent": ascii_safe("metaculus-ai-qgen/1.2"),
    }


def list_models_raw() -> List[Dict[str, Any]]:
    r = requests.get(OPENROUTER_MODELS, headers=or_headers(), timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("data") or data.get("models") or []


def list_models_clean() -> List[Dict[str, Any]]:
    try:
        ms = list_models_raw()
    except Exception:
        return []
    out = []
    for m in ms:
        out.append(
            {
                "id": m.get("id"),
                "name": m.get("name"),
                "context_length": m.get("context_length") or m.get("max_context_length"),
                "pricing": m.get("pricing") or {},
                "tags": m.get("tags") or [],
                "arch": m.get("architecture"),
            }
        )
    return out


def pick_model() -> str:
    if OPENROUTER_MODEL:
        return OPENROUTER_MODEL
    ms = list_models_clean()
    if ms:
        ids = {m.get("id"): m for m in ms if m.get("id")}
        for mid in PREFERRED_MODELS:
            if mid in ids:
                return mid
        best_id, best_price = None, 1e9
        for m in ms:
            mid = (m.get("id") or "").lower()
            tags = " ".join((m.get("tags") or [])).lower()
            arch = (m.get("arch") or "").lower()
            if ("instruct" in mid) or ("instruct" in tags) or ("instruct" in arch):
                pr = m.get("pricing") or {}
                p = pr.get("prompt") or pr.get("input") or 0.0
                try:
                    p = float(p) if p else 0.0
                except Exception:
                    p = 0.0
                if p < best_price:
                    best_price, best_id = p, m.get("id") or ""
        if best_id:
            return best_id
    return PREFERRED_MODELS[0]


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
                return s[start : i + 1]
    return None


def parse_json_relaxed(s: str, expect: str = "auto") -> Any:
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    fence = extract_code_fence(s)
    if fence:
        try:
            return json.loads(fence)
        except Exception:
            s = fence
    if expect in ("array", "auto"):
        blk = balanced_slice(s, "[", "]")
        if blk:
            try:
                return json.loads(blk)
            except Exception:
                pass
    blk = balanced_slice(s, "{", "}")
    if blk:
        try:
            return json.loads(blk)
        except Exception:
            pass
    objs = []
    for m in re.finditer(r"\{.*?\}", s, flags=re.DOTALL):
        try:
            objs.append(json.loads(m.group(0)))
        except Exception:
            continue
    if objs:
        return objs if len(objs) > 1 else objs[0]
    raise ValueError("Could not parse JSON from model output")


def call_openrouter(messages: List[Dict[str, str]], model: str, max_tokens: int = 2000, temperature: float = 0.2, retries: int = 3, expect: str = "auto") -> Any:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 1,
        "max_tokens": max_tokens,
    }
    last = None
    for k in range(retries):
        try:
            r = requests.post(OPENROUTER_URL, headers=or_headers(), json=payload, timeout=120)
            if r.status_code == 404:
                raise RuntimeError("404 No endpoints for model")
            if r.status_code == 429:
                retry_after = float(r.headers.get("Retry-After", "2") or 2)
                time.sleep(min(retry_after, 10))
                continue
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(str(data["error"]))
            ch = data.get("choices") or []
            if not ch:
                raise RuntimeError("No choices in response")
            content = ch[0].get("message", {}).get("content", "")
            if not content:
                raise RuntimeError("Empty content")
            return parse_json_relaxed(content, expect=expect)
        except Exception as e:
            last = e
            time.sleep(0.8 * (k + 1))
    raise RuntimeError(f"[openrouter] giving up after retries: {repr(last)}")


def load_examples_csv(path: str, k_good: int = 3, k_bad: int = 2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not path or not os.path.exists(path):
        return [], []
    goods, bads = [], []
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        rows = list(rd)

    def is_good(r: Dict[str, Any]) -> bool:
        t = (r.get("resolution_criteria") or r.get("resolution") or "").lower()
        return ("utc" in t or " by " in t or " on " in t) and ("will " in (r.get("title", "").lower()))

    def row2obj(r: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "title": r.get("title") or r.get("question_title") or "",
            "body": r.get("body") or r.get("background") or "",
            "resolution_criteria": r.get("resolution_criteria") or r.get("resolution") or "",
            "timeframe": r.get("timeframe") or r.get("end") or "",
            "answer_type": r.get("answer_type") or "",
        }

    random.shuffle(rows)
    for r in rows:
        o = row2obj(r)
        if not o["title"]:
            continue
        if is_good(r) and len(goods) < k_good:
            goods.append(o)
        elif not is_good(r) and len(bads) < k_bad:
            bads.append(o)
        if len(goods) >= k_good and len(bads) >= k_bad:
            break
    return goods, bads


GEN_SYS = "You are a senior Metaculus question writer. Return STRICT JSON only."
GEN_USER_TMPL = textwrap.dedent(
    """
    Task: Generate {n} candidate forecasting questions matching Metaculus style.

    Topic brief (3–6 lines):
    {brief}

    Domain tags: {tags} | Target horizon (if relevant): {horizon}

    For EACH candidate, output an object with:
    "title", "body", "resolution_criteria", "timeframe":{{"start":"...","end":"...","timezone":"UTC"}},
    "canonical_source": ["Publisher names or URLs allowed"], "answer_type": "binary|numeric|date|multiple",
    "proposed_bins_or_ranges": "...(if numeric)", "difficulty": "low|med|high",
    "rationale": "why decision-relevant and non-trivial", "policy_notes": "safety/legal notes".

    Constraints:
    - Outcomes must be independently verifiable from public sources; cite canonical_source (publishers allowed; URLs optional).
    - Include explicit end dates (UTC) and exact resolution checks; avoid vague terms unless thresholded.
    - If you have tools or internet access, research current figures/dates and cite the sources you rely on.
    - Title ≤ 100 chars; body 2–5 concise sentences.
    - Return a STRICT JSON array of {n} objects; no commentary, no markdown fences. If you add prose/fences, output will be discarded.

    Few-shot good examples:
    {good_examples}

    Few-shot bad/avoid examples (with reasons to avoid):
    {bad_examples}
    """
)

CRIT_SYS = "You are a meticulous Metaculus question editor. Return STRICT JSON."
CRIT_USER_TMPL = """Given this candidate JSON, rate 1–5 on each dimension:
clarity, falsifiability, operationalization, usefulness, safety.
List 3 concrete edits to raise any score <5. Then return a revised candidate.

Return:
{{
 "scores": {{...}},
 "edits": ["...","...","..."],
 "revised_candidate": {{...}}
}}

Candidate:
{candidate_json}
"""

JUDGE_SYS = "You are a strict Metaculus adjudicator. Return STRICT JSON only."
JUDGE_USER_TMPL = """Apply this rubric (1–5 each): clarity, falsifiability, operationalization, usefulness, safety.
Give overall (mean) and short notes. Return:
{{"scores":{{"clarity":int,"falsifiability":int,"operationalization":int,"usefulness":int,"safety":int}},"overall":X.X,"blockers":["..."],"notes":"..."}}

Candidate:
{candidate_json}
"""

PAIRWISE_SYS = "You are a strict adjudicator. Return STRICT JSON only."
PAIRWISE_USER_TMPL = """Compare Candidate A vs B for expected forecasting value to Metaculus users,
holding to the rubric. Pick a winner in {{"winner":"A"|"B","reason":"≤2 lines"}}.

A:
{A}

B:
{B}
"""


def _build_generation_prompt(brief: str, tags: List[str], horizon: str, n: int, good: List[Dict[str, Any]], bad: List[Dict[str, Any]]) -> str:
    good_str = json.dumps(good, ensure_ascii=False) if good else "[]"
    bad_str = json.dumps(bad, ensure_ascii=False) if bad else "[]"
    return GEN_USER_TMPL.format(
        n=n,
        brief=brief,
        tags=",".join(tags),
        horizon=horizon,
        good_examples=good_str,
        bad_examples=bad_str,
    )


def _mock_candidates(brief: str, n: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i in range(n):
        out.append(
            {
                "title": f"[MOCK] {brief[:40]} — Q{i+1}",
                "body": "Context lines. Why it matters. Actors involved.",
                "resolution_criteria": "On 2030-12-31 23:59:59 UTC, check source X for Y; YES if Z; otherwise NO.",
                "timeframe": {
                    "start": "2026-01-01 00:00:00",
                    "end": "2030-12-31 23:59:59",
                    "timezone": "UTC",
                },
                "canonical_source": ["Reuters", "official press release"],
                "answer_type": "binary",
                "proposed_bins_or_ranges": "",
                "difficulty": "med",
                "rationale": "Decision-relevant; not trivially predictable.",
                "policy_notes": "",
            }
        )
    return out


def generate_candidates(brief: str, tags: List[str], horizon: str, n: int, good: List[Dict[str, Any]], bad: List[Dict[str, Any]], model: str, dry_run: bool = False) -> List[Dict[str, Any]]:
    if dry_run:
        return _mock_candidates(brief, n)

    user = _build_generation_prompt(brief=brief, tags=tags, horizon=horizon, n=n, good=good, bad=bad)
    resp = call_openrouter(
        [{"role": "system", "content": GEN_SYS}, {"role": "user", "content": user}],
        model=model,
        max_tokens=4000,
        temperature=0.5,
        expect="array",
    )
    if isinstance(resp, list):
        return resp
    if isinstance(resp, dict) and "candidates" in resp and isinstance(resp["candidates"], list):
        return resp["candidates"]
    if isinstance(resp, dict):
        return [resp]
    raise RuntimeError("Generation returned unexpected shape")


def critique_and_revise(cand: Dict[str, Any], model: str, dry_run: bool = False) -> Tuple[Dict[str, Any], Dict[str, int]]:
    if dry_run:
        return cand, {
            "clarity": 4,
            "falsifiability": 4,
            "operationalization": 4,
            "usefulness": 4,
            "safety": 5,
        }
    user = CRIT_USER_TMPL.format(candidate_json=json.dumps(cand, ensure_ascii=False))
    resp = call_openrouter(
        [{"role": "system", "content": CRIT_SYS}, {"role": "user", "content": user}],
        model=model,
        max_tokens=2000,
        temperature=0.1,
        expect="object",
    )
    revised = resp.get("revised_candidate") or cand
    scores = {k.lower(): int(round(float(v))) for k, v in (resp.get("scores") or {}).items()}
    return revised, scores


def judge(cand: Dict[str, Any], model: str, dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        base = 3.6 + random.random() * 1.0
        return {
            "scores": {
                "clarity": 4,
                "falsifiability": 4,
                "operationalization": 4,
                "usefulness": 4,
                "safety": 5,
            },
            "overall": round(min(5.0, base), 2),
            "blockers": [],
            "notes": "mock",
        }
    user = JUDGE_USER_TMPL.format(candidate_json=json.dumps(cand, ensure_ascii=False))
    resp = call_openrouter(
        [{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": user}],
        model=model,
        max_tokens=1200,
        temperature=0.0,
        expect="object",
    )
    resp["overall"] = float(resp.get("overall", 0.0))
    return resp


def pairwise_battle(A: Dict[str, Any], B: Dict[str, Any], model: str, dry_run: bool = False) -> str:
    if dry_run:
        return "A" if random.random() < 0.5 else "B"
    user = PAIRWISE_USER_TMPL.format(A=json.dumps(A, ensure_ascii=False), B=json.dumps(B, ensure_ascii=False))
    resp = call_openrouter(
        [{"role": "system", "content": PAIRWISE_SYS}, {"role": "user", "content": user}],
        model=model,
        max_tokens=400,
        temperature=0.0,
        expect="object",
    )
    return resp.get("winner", "A")


def run_pipeline_in_memory(brief: str, tags: List[str], horizon: str, n: int = 10, examples_csv: Optional[str] = None, top_k: int = 5, dry_run: bool = False) -> Dict[str, Any]:
    gen_model = pick_model()
    judge_model = pick_model()
    crit_model = pick_model()
    good, bad = load_examples_csv(examples_csv or "", k_good=3, k_bad=2)
    cands = generate_candidates(brief, tags, horizon, n, good, bad, gen_model, dry_run=dry_run)
    revised = []
    crit_scores = []
    for c in cands:
        r, s = critique_and_revise(c, crit_model, dry_run=dry_run)
        revised.append(r)
        crit_scores.append(s)
    judgements = [judge(c, judge_model, dry_run=dry_run) for c in revised]
    idx = sorted(range(len(judgements)), key=lambda i: -judgements[i].get("overall", 0.0))[: max(2, top_k)]
    top = [revised[i] for i in idx]
    top_scores = [judgements[i] for i in idx]
    wins = {i: 0 for i in range(len(top))}
    for i, j in itertools.combinations(range(len(top)), 2):
        w = pairwise_battle(top[i], top[j], judge_model, dry_run=dry_run)
        if w == "A":
            wins[i] += 1
        elif w == "B":
            wins[j] += 1
    ranked = [x for _, x in sorted(((wins[i], i) for i in range(len(top))), reverse=True)]
    top = [top[i] for i in ranked]
    top_scores = [top_scores[i] for i in ranked]
    return {
        "gen_model": gen_model,
        "crit_model": crit_model,
        "judge_model": judge_model,
        "candidates": top,
        "judgements": top_scores,
    }


st.set_page_config(page_title="Metaculus AI Question Generator", page_icon="📊", layout="wide")

st.title("Metaculus AI Question Generation – Panel v1")

st.markdown(
    """
This app wraps a Metaculus-style AI question generation pipeline
around the OpenRouter API, with critique, judging and pairwise ranking.
"""
)

with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input("OpenRouter API key", type="password", help="Key will be kept only in this session.")
    if api_key_input:
        st.session_state["OPENROUTER_API_KEY_OVERRIDE"] = api_key_input.strip()
    n = st.slider("Number of candidates to generate", min_value=3, max_value=30, value=10, step=1)
    top_k = st.slider("Top K after ranking", min_value=2, max_value=10, value=5, step=1)
    dry_run = st.checkbox("Dry run (no API calls, mock output)", value=False)
    scrape_n = st.slider("Scrape N Metaculus questions for few-shot examples (0 = none)", min_value=0, max_value=50, value=0, step=5)
    examples_file = st.file_uploader("Or upload Metaculus example questions CSV", type=["csv"])

current_key = get_openrouter_key()
if not current_key and not dry_run:
    st.warning("No OPENROUTER_API_KEY detected. Enter it in the sidebar, or set it as an environment variable / Streamlit secret.")

st.subheader("Problem setup")

brief = st.text_area(
    "Topic brief (3–6 lines)",
    height=150,
    placeholder="e.g. medium-term AI capability benchmarks, regulation in the EU/US, deployment race dynamics...",
)

tags_str = st.text_input("Domain tags (comma-separated)", value="ai,policy,geopolitics")

horizon = st.text_input("Horizon / resolution description", value="resolve by 2030-12-31 UTC")

run_button = st.button("Generate and rank questions")

if run_button:
    current_key = get_openrouter_key()
    if not brief.strip():
        st.warning("Please provide at least a short topic brief.")
    elif not current_key and not dry_run:
        st.error("No OPENROUTER_API_KEY set and dry_run is disabled.")
    else:
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        examples_path = None
        if scrape_n > 0:
            try:
                examples_path = scrape_metaculus_examples_to_csv(scrape_n)
            except Exception as e:
                st.error(f"Metaculus scraping error: {e}")
                st.stop()
        elif examples_file is not None:
            fd, path = tempfile.mkstemp(suffix=".csv")
            with os.fdopen(fd, "wb") as f:
                f.write(examples_file.read())
            examples_path = path
        with st.spinner("Running generation, critique, judging and ranking..."):
            try:
                res = run_pipeline_in_memory(
                    brief=brief,
                    tags=tags,
                    horizon=horizon,
                    n=n,
                    examples_csv=examples_path,
                    top_k=top_k,
                    dry_run=dry_run,
                )
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                raise
        st.success("Done")
        gen_model = res["gen_model"]
        crit_model = res["crit_model"]
        judge_model = res["judge_model"]
        cands = res["candidates"]
        judgements = res["judgements"]
        st.subheader("Models used")
        col1, col2, col3 = st.columns(3)
        col1.metric("Generator model", gen_model)
        col2.metric("Critique model", crit_model)
        col3.metric("Judge model", judge_model)
        rows = []
        for c, s in zip(cands, judgements):
            row = {
                "overall": s.get("overall", 0.0),
                "clarity": (s.get("scores") or {}).get("clarity"),
                "falsifiability": (s.get("scores") or {}).get("falsifiability"),
                "operationalization": (s.get("scores") or {}).get("operationalization"),
                "usefulness": (s.get("scores") or {}).get("usefulness"),
                "safety": (s.get("scores") or {}).get("safety"),
                "title": c.get("title", ""),
                "body": c.get("body", ""),
                "resolution_criteria": c.get("resolution_criteria", ""),
                "timeframe_start": (c.get("timeframe") or {}).get("start", ""),
                "timeframe_end": (c.get("timeframe") or {}).get("end", ""),
                "timezone": (c.get("timeframe") or {}).get("timezone", ""),
                "answer_type": c.get("answer_type", ""),
                "proposed_bins_or_ranges": c.get("proposed_bins_or_ranges", ""),
                "canonical_source": "; ".join(c.get("canonical_source") or []),
                "difficulty": c.get("difficulty", ""),
                "rationale": c.get("rationale", ""),
                "policy_notes": c.get("policy_notes", ""),
                "judge_notes": s.get("notes", ""),
            }
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            df = df.sort_values("overall", ascending=False)
            st.subheader("Ranked candidate questions")
            st.dataframe(df, use_container_width=True, height=500)
            top_row = df.iloc[0]
            st.markdown("### Top candidate")
            st.markdown(f"**Title:** {top_row['title']}")
            st.markdown(f"**Body:** {top_row['body']}")
            st.markdown(f"**Resolution criteria:** {top_row['resolution_criteria']}")
            st.markdown(
                f"**Timeframe:** {top_row['timeframe_start']} → {top_row['timeframe_end']} ({top_row.get('timezone','UTC')})"
            )
            st.markdown(f"**Answer type:** {top_row['answer_type']}")
            st.markdown(f"**Rationale:** {top_row['rationale']}")
            if top_row.get("policy_notes"):
                st.markdown(f"**Policy notes:** {top_row['policy_notes']}")
            if top_row.get("judge_notes"):
                st.markdown(f"**Judge notes:** {top_row['judge_notes']}")
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            csv_bytes = csv_buf.getvalue().encode("utf-8")
            records = []
            for c, s in zip(cands, judgements):
                records.append({"candidate": c, "judge": s})
            jsonl_buf = io.StringIO()
            for r in records:
                jsonl_buf.write(json.dumps(r, ensure_ascii=False) + "\n")
            jsonl_bytes = jsonl_buf.getvalue().encode("utf-8")
            st.subheader("Download")
            c1, c2 = st.columns(2)
            c1.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name="metaculus_ai_qgen_top.csv",
                mime="text/csv",
            )
            c2.download_button(
                "Download JSONL",
                data=jsonl_bytes,
                file_name="metaculus_ai_qgen_top.jsonl",
                mime="application/json",
            )
        else:
            st.info("No candidates generated.")

