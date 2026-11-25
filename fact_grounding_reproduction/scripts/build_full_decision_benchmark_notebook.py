"""Build Notebook 05: Full 8-Benchmark Decision-Model Evaluation for Fact-Grounding DeBERTa.

Designed for Kaggle GPU (T4 x2 / T4 / P100) under account `huthnhnguyn` using the
already-uploaded dataset `huthnhnguyn/fact-grounding-deberta-v3-large-checkpoint`.
All 8 benchmarks (9 evaluation tracks) download automatically from pinned public
GitHub / Hugging Face sources with zero additional dataset uploads required.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "kaggle_full_decision_eval"

INTRO_MD = r"""# Fact-Grounding DeBERTa-v3-Large: Comprehensive 8-Benchmark Decision Evaluation

This notebook evaluates the **existing fine-tuned DeBERTa-v3-large fact-grounding checkpoint** (`huthnhnguyn/fact-grounding-deberta-v3-large-checkpoint`) strictly as a **binary evidence-sufficiency decision model** (`0 = NOT-ANSWERABLE`, `1 = ANSWERABLE`) across **8 benchmarks (9 tracks)** spanning in-domain multi-hop, native unanswerable multi-hop, incomplete-information multi-hop, cross-domain news, counterfactual/unanswerable faithfulness, and single-hop answerability control.

It does **not** train, fine-tune, or tune thresholds on any OOD target set.

## Evaluation Tracks

| Track | Benchmark | Label Type | Format | What It Tests |
|---|---|---|---|---|
| `01_musique_dev` | MuSiQue dev (official paper CSV, 6,404 hops) | Official paper counterfactual hop labels | Sub-question + passages | In-domain reference (note: used for checkpoint selection) |
| `02a_musique_full_q` | MuSiQue-Full dev (4,834 questions, 50/50) | Official native `answerable` boolean | Full multi-hop question + passages | Full-question format shift on same MuSiQue corpus |
| `02b_musique_full_hops` | MuSiQue-Full dev decomposed hops | Official `paragraph_support_idx != null` | Sub-question + passages | Native unanswerable hops in exact paper sub-question format |
| `03_iirc_dev` | IIRC dev (1,301 questions, EMNLP 2020) | Official native `answer.type != "none"` | Multi-hop incomplete-info question | Native missing-link / incomplete evidence multi-hop QA |
| `04_hotpotqa_pairs` | HotpotQA distractor dev (~9,088 rows) | Controlled support-fact ablation pairs | Full question + supporting/ablated facts | Transfer to 2-hop Wikipedia QA (matches v5 protocol) |
| `05_2wiki_support_pairs` | 2WikiMultiHopQA dev (~7,188 rows) | Controlled support-deletion pairs | Full question + supporting/ablated facts | Relational Wikipedia multi-hop sensitivity (matches v5 protocol) |
| `06_faitheval_pairs` | FaithEval Unanswerable (ICLR 2025, 4,982 rows) | Official original vs edited unanswerable pairs | Full question + original/edited context | Context faithfulness & unanswerable detection (matches v5 protocol) |
| `07_multihop_rag_news` | MultiHop-RAG (COLM 2024, 2,556 queries) | Native `null_query` vs answerable queries | News multi-hop query + evidence/BM25 news | Cross-domain transfer off Wikipedia to real news articles |
| `08_squad_v2_dev` | SQuAD 2.0 dev (11,873 examples, 50/50) | Official native answerable/unanswerable | Single-hop question + paragraph | Single-hop control with clear DeBERTa-v3-large SOTA anchor |

## Protocol & Calibration

- Exact paper input format: `Sub-question: {question}\n\nPassages:\n{passages[:1500]}`, max sequence length `512`, decision threshold `0.5`.
- Saves raw `logit_diff = logit_1 - logit_0` per row to compute both:
  1. **Raw paper protocol (`threshold = 0.5`)** in `metrics.csv`.
  2. **Post-hoc Temperature Scaling + F1-optimal threshold** fit **strictly on `01_musique_dev`** and frozen across all OOD tracks (`02a`–`08`) in `metrics_calibrated.csv` (zero target-domain leakage).
- Durable batch JSONL logging + atomic `progress_state.json` + periodic `resume_bundle.zip`.
"""

SETUP_CELL = r"""import sys, subprocess, time
NOTEBOOK_STARTED_AT = time.monotonic()
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.57.1", "sentencepiece", "protobuf", "scikit-learn", "pyarrow", "scipy"
])
print("Runtime setup complete. Wall-time budget: 8.0 hours from this cell.")
"""

MAIN_CELL = r'''import os, sys, re, json, time, math, hashlib, random, shutil, zipfile, warnings
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import torch
from scipy.optimize import minimize_scalar
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, balanced_accuracy_score, roc_auc_score,
    average_precision_score, brier_score_loss
)
from IPython.display import display, FileLink

warnings.filterwarnings("ignore")

# ----- Fixed experiment configuration -----
MAX_RUNTIME_HOURS = 8.0
MAX_LENGTH = 512
PASSAGE_MAX_CHARS = 1500
DECISION_THRESHOLD = 0.5
BASE_BATCH_SIZE = 32
SEED = 42
EXPECTED_LOCAL_SHA256 = "93EF195F3840689A70094C8FDBE4C1126A86E9FAF1B9382A3040E3D32B470260"
CHECKPOINT_DATASET_HINT = "fact-grounding-deberta-v3-large-checkpoint"
STARTED_AT = globals().get("NOTEBOOK_STARTED_AT", time.monotonic())

INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working/fact_grounding_8_benchmark_eval")
PREDICTION_ROOT = OUTPUT_ROOT / "predictions"
CACHE_ROOT = OUTPUT_ROOT / "download_cache"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
PREDICTION_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUTPUT_ROOT / "run_log.jsonl"
STATE_PATH = OUTPUT_ROOT / "progress_state.json"

print("Initial /kaggle/input entries:", [p.name for p in INPUT_ROOT.iterdir()] if INPUT_ROOT.exists() else "<missing>")

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def log_event(event, **fields):
    record = {"time_utc": utc_now(), "event": event, **fields}
    line = json.dumps(record, ensure_ascii=False)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush(); os.fsync(f.fileno())
    print("[RUNLOG] " + line, flush=True)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def safe_read_jsonl(path):
    rows = []
    if not Path(path).exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows

def elapsed_hours():
    return (time.monotonic() - STARTED_AT) / 3600.0

def check_time_budget():
    return elapsed_hours() < MAX_RUNTIME_HOURS

def restore_resume_bundle():
    restored = []
    if not INPUT_ROOT.exists():
        return
    for bundle in INPUT_ROOT.rglob("resume_bundle.zip"):
        try:
            with zipfile.ZipFile(bundle) as zf:
                for name in zf.namelist():
                    rel = Path(name)
                    if rel.is_absolute() or ".." in rel.parts:
                        continue
                    target = (OUTPUT_ROOT / rel).resolve()
                    if OUTPUT_ROOT.resolve() not in target.parents and target != OUTPUT_ROOT.resolve():
                        continue
                    if target.exists():
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(name) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    restored.append(str(rel))
        except Exception as exc:
            print(f"Could not restore {bundle}: {exc}")
    if restored:
        print("Restored resume files:", restored[:12], "..." if len(restored) > 12 else "")

def write_resume_bundle():
    tmp = OUTPUT_ROOT / "resume_bundle.tmp"
    final = OUTPUT_ROOT / "resume_bundle.zip"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
        for p in OUTPUT_ROOT.rglob("*"):
            if not p.is_file() or p.name in {"resume_bundle.zip", "resume_bundle.tmp"}:
                continue
            if CACHE_ROOT in p.parents:
                continue
            zf.write(p, p.relative_to(OUTPUT_ROOT).as_posix())
    os.replace(tmp, final)

restore_resume_bundle()
log_event("run_started", time_budget_hours=MAX_RUNTIME_HOURS, output_dir=str(OUTPUT_ROOT))
state = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {"benchmarks": {}}
state.setdefault("benchmarks", {})

# ----- Locate & verify attached checkpoint -----
def find_checkpoint_file():
    candidates = []
    if INPUT_ROOT.exists():
        for p in INPUT_ROOT.rglob("model.safetensors"):
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(
            f"No model.safetensors found under {INPUT_ROOT}. "
            f"Attach Kaggle dataset '{CHECKPOINT_DATASET_HINT}' before running."
        )
    # Prefer path matching hint if multiple datasets are attached.
    for p in candidates:
        if CHECKPOINT_DATASET_HINT in str(p).lower() or "fact-grounding" in str(p).lower():
            return p
    return candidates[0]

checkpoint_file = find_checkpoint_file()
checkpoint_dir = checkpoint_file.parent
checkpoint_sha256 = sha256_file(checkpoint_file)
cfg_path = checkpoint_dir / "config.json"
ckpt_config = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
print("Checkpoint file:", checkpoint_file)
print("Checkpoint bytes:", f"{checkpoint_file.stat().st_size:,}")
print("Checkpoint SHA-256:", checkpoint_sha256)
print("Matches expected local SHA-256:", checkpoint_sha256.upper() == EXPECTED_LOCAL_SHA256)
print("Config id2label:", ckpt_config.get("id2label"))
if ckpt_config.get("id2label") != {"0": "NOT-ANSWERABLE", "1": "ANSWERABLE"}:
    raise ValueError(f"Unexpected checkpoint id2label mapping: {ckpt_config.get('id2label')}")
log_event(
    "checkpoint_verified",
    file=str(checkpoint_file),
    bytes=checkpoint_file.stat().st_size,
    sha256=checkpoint_sha256,
    matches_expected_local_sha256=(checkpoint_sha256.upper() == EXPECTED_LOCAL_SHA256),
    id2label=ckpt_config.get("id2label"),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)] if n_gpus else ["CPU"]
print(f"Device: {device} | n_gpus={n_gpus} | {gpu_names}")

tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
base_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir, local_files_only=True)
base_model.to(device).eval()
if n_gpus > 1:
    model = torch.nn.DataParallel(base_model)
    effective_batch_size = BASE_BATCH_SIZE * n_gpus
else:
    model = base_model
    effective_batch_size = BASE_BATCH_SIZE

log_event(
    "model_loaded",
    device=str(device),
    n_gpus=n_gpus,
    gpu_names=gpu_names,
    effective_batch_size=effective_batch_size,
    max_length=MAX_LENGTH,
    threshold=DECISION_THRESHOLD,
)

# ----- Shared text & retrieval helpers -----
def normalize_text(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").casefold()).strip()

def serialize_input(question, passage):
    if isinstance(passage, (list, tuple)):
        passage = "\n".join(str(p) for p in passage)
    passage_raw = str(passage or "")
    was_truncated = len(passage_raw) > PASSAGE_MAX_CHARS
    passage_cap = passage_raw[:PASSAGE_MAX_CHARS]
    return f"Sub-question: {str(question or '').strip()}\n\nPassages:\n{passage_cap}", was_truncated

def canonical_text(row):
    return str(row["input_text"])

def rows_signature(rows):
    h = hashlib.sha256()
    for row in rows:
        payload = [str(row["example_id"]), int(row["label"]), canonical_text(row)]
        h.update(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()

def read_jsonish(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value

def download_public_file(url, destination, retries=4):
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    tmp = destination.with_suffix(destination.suffix + ".part")
    last_error = None
    for attempt in range(retries):
        if not check_time_budget():
            raise TimeoutError("Runtime budget reached before data download completed.")
        try:
            with requests.get(url, stream=True, timeout=(30, 180), headers={"User-Agent": "fact-grounding-eval/2.0"}) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                    f.flush(); os.fsync(f.fileno())
            os.replace(tmp, destination)
            return destination
        except Exception as exc:
            last_error = exc
            if tmp.exists():
                tmp.unlink()
            time.sleep(min(3 * (attempt + 1), 12))
    raise RuntimeError(f"Download failed after {retries} attempts: {url}") from last_error

# Lightweight pure-Python BM25 for small per-question / 609-article corpora (zero extra deps)
STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "is", "are", "was", "were",
    "be", "by", "at", "from", "with", "that", "this", "as", "it", "what", "which", "who", "when",
    "where", "how", "did", "do", "does", "has", "have", "had", "not", "but", "about", "into"
}

def tokenize_bm25(text):
    return [t for t in normalize_text(text).split() if len(t) > 1 and t not in STOPWORDS]

class SimpleBM25:
    def __init__(self, docs_texts, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.N = len(docs_texts)
        self.doc_tfs = []
        self.doc_lens = []
        df = Counter()
        for txt in docs_texts:
            toks = tokenize_bm25(txt)
            self.doc_lens.append(len(toks))
            tf = Counter(toks)
            self.doc_tfs.append(tf)
            for term in tf:
                df[term] += 1
        self.avgdl = (sum(self.doc_lens) / self.N) if self.N else 1.0
        self.idf = {t: math.log(1.0 + (self.N - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def top_k(self, query, k=3):
        q_toks = tokenize_bm25(query)
        scores = []
        for idx, (tf, dl) in enumerate(zip(self.doc_tfs, self.doc_lens)):
            s = 0.0
            for t in q_toks:
                if t in tf:
                    idf = self.idf.get(t, 0.0)
                    f = tf[t]
                    s += idf * (f * (self.k1 + 1.0)) / (f + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl))
            scores.append((s, -idx, idx))
        scores.sort(reverse=True)
        return [idx for _, _, idx in scores[:k]]

# =====================================================================
# BENCHMARK BUILDERS (8 Benchmarks / 9 Tracks)
# =====================================================================
benchmarks = {}
benchmark_sources = {}

# ----- Track 01: MuSiQue official paper dev CSV (6,404 hop rows) -----
b1_url = "https://raw.githubusercontent.com/RichardZhu123/fact-grounding-gap/3df77c2c94863ff35b398e14b37bd1e130e9d6de/deberta_fg_dev.csv"
b1_path = download_public_file(b1_url, CACHE_ROOT / "deberta_fg_dev.csv")
dev_df = pd.read_csv(b1_path)
b1_rows = []
for idx, r in dev_df.iterrows():
    raw_text = str(r["text"])
    b1_rows.append({
        "example_id": f"musique-dev-{idx}", "pair_id": None,
        "question": raw_text.split("\n\nPassages:\n", 1)[0].replace("Sub-question: ", "").strip(),
        "input_text": raw_text, "char_truncated": False,
        "label": int(r["label"]),
        "label_source": "Official released MuSiQue dev counterfactual hop label",
        "construction_method": "Exact paper dev CSV serialization."
    })
benchmarks["01_musique_dev"] = b1_rows
benchmark_sources["01_musique_dev"] = {
    "url": b1_url, "sha256": sha256_file(b1_path), "n_examples": len(b1_rows),
    "caveat": "In-domain MuSiQue dev split used for checkpoint selection."
}

# ----- Track 02a & 02b: MuSiQue-Full dev (4,834 native answerable/unanswerable) -----
b2_url = "https://huggingface.co/datasets/bdsaglam/musique/resolve/main/musique_full_v1.0_dev.jsonl?download=true"
b2_path = download_public_file(b2_url, CACHE_ROOT / "musique_full_v1.0_dev.jsonl")
mf_items = [json.loads(line) for line in b2_path.read_text(encoding="utf-8").splitlines() if line.strip()]

b2a_rows = []
b2b_rows = []
for mf_idx, item in enumerate(mf_items):
    qid = str(item["id"])
    ans_tag = "ans" if bool(item.get("answerable")) else "unans"
    question = str(item.get("question", "")).strip()
    answerable = bool(item.get("answerable"))
    paragraphs = item.get("paragraphs", [])
    para_by_idx = {int(p["idx"]): p for p in paragraphs}
    para_texts = [f"[{p.get('title', '')}] {p.get('paragraph_text', '').strip()}" for p in paragraphs]
    bm25_q = SimpleBM25(para_texts)

    # 02a: Full-question view with realistic top-3 passage budget fitting inside 1,500 chars.
    # For answerable=True: include all official supporting paragraphs (2-4); fill to 3 if fewer.
    # For answerable=False: include any marked supporting paragraphs + top-BM25 distractors to match k=3.
    supp_indices = [i for i, p in enumerate(paragraphs) if p.get("is_supporting")]
    if answerable and supp_indices:
        chosen_indices = list(supp_indices)
        if len(chosen_indices) < 3:
            for bi in bm25_q.top_k(question, k=5):
                if bi not in chosen_indices:
                    chosen_indices.append(bi)
                if len(chosen_indices) >= 3:
                    break
    else:
        chosen_indices = list(supp_indices)
        for bi in bm25_q.top_k(question, k=6):
            if bi not in chosen_indices:
                chosen_indices.append(bi)
            if len(chosen_indices) >= 3:
                break
    # Keep deterministic original paragraph order so position bias does not leak labels.
    chosen_indices = sorted(set(chosen_indices))[:4]
    passage_str = "\n".join(para_texts[i] for i in chosen_indices)
    inp, trunc = serialize_input(question, passage_str)
    b2a_rows.append({
        "example_id": f"musique-full-q-{mf_idx}-{qid}-{ans_tag}", "pair_id": f"musique-full-{qid}",
        "question": question, "input_text": inp, "char_truncated": trunc,
        "label": 1 if answerable else 0,
        "label_source": "Official MuSiQue-Full native answerable boolean",
        "construction_method": "Full multi-hop question with supporting/BM25 top-3 paragraphs (sorted by paragraph idx)."
    })

    # 02b: Decomposed hop-level sub-question view (exact match to paper sub-question format!).
    decomp = item.get("question_decomposition", [])
    prev_answers = {}
    for hop_i, step in enumerate(decomp):
        sub_q = str(step.get("question", "")).strip()
        for prev_idx, prev_ans in prev_answers.items():
            sub_q = sub_q.replace(f"#{prev_idx + 1}", prev_ans)
        if step.get("answer"):
            prev_answers[hop_i] = str(step["answer"]).strip()
        supp_idx = step.get("paragraph_support_idx")
        hop_label = 1 if (supp_idx is not None and int(supp_idx) in para_by_idx) else 0
        if hop_label == 1:
            gold_p = para_by_idx[int(supp_idx)]
            gold_text = f"[{gold_p.get('title', '')}] {gold_p.get('paragraph_text', '').strip()}"
            # Add 2 BM25 distractors to mirror k=3 hop retrieval context
            hop_cands = [int(supp_idx)]
            for bi in bm25_q.top_k(sub_q, k=5):
                p_idx = int(paragraphs[bi]["idx"])
                if p_idx not in hop_cands:
                    hop_cands.append(p_idx)
                if len(hop_cands) >= 3:
                    break
            hop_cands.sort()
            hop_passages = "\n".join(
                f"[{para_by_idx[i].get('title', '')}] {para_by_idx[i].get('paragraph_text', '').strip()}"
                for i in hop_cands
            )
        else:
            top_b = sorted(bm25_q.top_k(sub_q, k=3))
            hop_passages = "\n".join(para_texts[i] for i in top_b)
        inp_h, trunc_h = serialize_input(sub_q, hop_passages)
        b2b_rows.append({
            "example_id": f"musique-full-hop-{mf_idx}-{qid}-{ans_tag}-h{hop_i}", "pair_id": None,
            "question": sub_q, "input_text": inp_h, "char_truncated": trunc_h,
            "label": hop_label,
            "label_source": "Official MuSiQue-Full hop paragraph_support_idx != null",
            "construction_method": "Resolved decomposed sub-question + k=3 passages (gold+BM25 if supported, else BM25 top-3)."
        })

benchmarks["02a_musique_full_q"] = b2a_rows
benchmark_sources["02a_musique_full_q"] = {
    "url": b2_url, "sha256": sha256_file(b2_path), "n_examples": len(b2a_rows),
    "caveat": "Native MuSiQue-Full dev answerability (full-question format shift vs training sub-questions)."
}
benchmarks["02b_musique_full_hops"] = b2b_rows
benchmark_sources["02b_musique_full_hops"] = {
    "url": b2_url, "sha256": sha256_file(b2_path), "n_examples": len(b2b_rows),
    "caveat": "Decomposed hop-level sub-questions from MuSiQue-Full dev (exact paper sub-question format; unbalanced ~66% pos / ~34% neg)."
}

# ----- Track 03: IIRC dev (1,301 native incomplete-info multi-hop questions) -----
b3_url = "https://huggingface.co/datasets/voidful/IIRC/resolve/main/dev.json?download=true"
b3_path = download_public_file(b3_url, CACHE_ROOT / "iirc_dev.json")
iirc_data = json.loads(b3_path.read_text(encoding="utf-8"))
b3_rows = []
for p_idx, item in enumerate(iirc_data):
    main_title = str(item.get("title", "")).strip()
    main_text = str(item.get("text", "")).strip()
    pid = str(item.get("pid", p_idx))
    for q_idx, qobj in enumerate(item.get("questions", [])):
        q_text = str(qobj.get("question", "")).strip()
        ans_type = str(qobj.get("answer", {}).get("type", "")).strip().lower()
        if not q_text or not ans_type:
            continue
        label = 0 if ans_type == "none" else 1
        ctx_items = qobj.get("context", [])
        parts = [f"[{main_title}] {main_text}"]
        seen_texts = {main_text}
        for c in ctx_items:
            c_pass = str(c.get("passage", "")).strip()
            c_txt = str(c.get("text", "")).strip()
            if c_txt and c_pass.lower() != "main" and c_txt not in seen_texts:
                parts.append(f"[{c_pass}] {c_txt}")
                seen_texts.add(c_txt)
        inp, trunc = serialize_input(q_text, parts)
        b3_rows.append({
            "example_id": f"iirc-dev-{pid}-q{q_idx}", "pair_id": None,
            "question": q_text, "input_text": inp, "char_truncated": trunc,
            "label": label,
            "label_source": f"Official IIRC dev native answer.type ({ans_type})",
            "construction_method": "Main Wikipedia paragraph + annotated linked context snippets; label=0 iff answer.type=='none'."
        })
benchmarks["03_iirc_dev"] = b3_rows
benchmark_sources["03_iirc_dev"] = {
    "url": b3_url, "sha256": sha256_file(b3_path), "n_examples": len(b3_rows),
    "caveat": "Native incomplete-information multi-hop QA (EMNLP 2020); full-question format; ~26.7% unanswerable."
}

# ----- Track 04: HotpotQA distractor dev pairs (exact parity with prepare_eval_data.py) -----
MANUAL_EXCLUSIONS = {
    "5a78b386554299148911f949", "5a7e31c055429934daa2fbee", "5ae3fd2c5542995dadf2428f",
    "5ae3790f5542992e3233c415", "5adf72d35542992d7e9f9369", "5abfa2255542993fe9a41e18",
    "5ac311925542995ef918c0de", "5ac1a4ed5542994d76dcce90", "5a8230af554299676cceb227",
    "5a851b0d5542997175ce1f7f", "5a886098554299206df2b248", "5a79c8945542994bb945709e",
    "5add9bfa5542990dbb2f7e85", "5ade6acf554299728e26c71a", "5a8671e95542994775f60763",
    "5add98985542990dbb2f7e82", "5ae074aa5542993d6555eb54", "5abbc4f955429931dba144ff",
    "5ae0d5335542990adbacf674", "5a7af8fa55429927d897bf25", "5a84b7db5542991dd0999db0",
    "5a79c2955542994f819ef097", "5a73987c55429978a71e9039", "5a78f89055429974737f792b",
    "5a8f155e554299458435d54c", "5a8481ba5542997175ce1ed4", "5a8fab8c5542995b4424208a",
    "5ae056c755429945ae959319", "5ae395d85542992f92d82313", "5ae277575542996483e649b2",
    "5ae5df84554299546bf82fa8", "5ac26eed55429951e9e685bf", "5aba94e355429901930fa845",
    "5ac560235542993e66e82396", "5ac0dcf25542996f0d89cc2c", "5ae54e005542993aec5ec19a",
}

def as_list(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def context_documents(value):
    value = read_jsonish(value)
    docs = []
    if isinstance(value, dict):
        titles = as_list(value.get("title") if value.get("title") is not None else value.get("titles"))
        sentences = as_list(value.get("sentences") if value.get("sentences") is not None else value.get("sentence"))
        for title, sents in zip(titles, sentences):
            docs.append((str(title), [str(s) for s in as_list(sents)]))
    elif isinstance(value, (list, tuple, np.ndarray)):
        for item in as_list(value):
            item = read_jsonish(item)
            if isinstance(item, dict):
                raw_s = item.get("sentences") if item.get("sentences") is not None else item.get("sentence")
                docs.append((str(item.get("title", "")), [str(s) for s in as_list(raw_s)]))
            elif isinstance(item, (list, tuple, np.ndarray)):
                item_l = as_list(item)
                if len(item_l) >= 2:
                    sents = read_jsonish(item_l[1])
                    if isinstance(sents, str):
                        sents = [sents]
                    docs.append((str(item_l[0]), [str(s) for s in as_list(sents)]))
    return docs

def support_tuples(value):
    value = read_jsonish(value)
    facts = []
    if isinstance(value, dict):
        titles = as_list(value.get("title") if value.get("title") is not None else value.get("titles"))
        raw_idx = value.get("sent_id")
        if raw_idx is None:
            raw_idx = value.get("sentence_id")
        if raw_idx is None:
            raw_idx = value.get("sent_ids")
        indices = as_list(raw_idx)
        for title, idx in zip(titles, indices):
            try: facts.append((str(title), int(idx)))
            except Exception: pass
    elif isinstance(value, (list, tuple, np.ndarray)):
        for item in as_list(value):
            if isinstance(item, (list, tuple, np.ndarray)):
                item_l = as_list(item)
                if len(item_l) >= 2:
                    try: facts.append((str(item_l[0]), int(item_l[1])))
                    except Exception: pass
            elif isinstance(item, dict):
                try: facts.append((str(item.get("title", "")), int(item.get("sent_id", item.get("sentence_id")))))
                except Exception: pass
    return facts

b4_url = "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet?download=true"
b4_path = download_public_file(b4_url, CACHE_ROOT / "hotpotqa_distractor_val.parquet")
hp_df = pd.read_parquet(b4_path)
b4_rows = []
for ex in hp_df.to_dict("records"):
    group_id = str(ex.get("id", ex.get("_id", "")))
    if group_id in MANUAL_EXCLUSIONS:
        continue
    facts = list(dict.fromkeys(support_tuples(ex.get("supporting_facts"))))
    if len(facts) < 2:
        continue
    docs = context_documents(ex.get("context"))
    contexts = {title: sents for title, sents in docs}
    resolved = []
    valid = True
    for title, sid in facts:
        sents = contexts.get(title)
        if sents is None or sid < 0 or sid >= len(sents):
            valid = False
            break
        resolved.append((title, sid, sents[sid].strip()))
    if not valid or any(not s for _, _, s in resolved):
        continue
    answer = str(ex.get("answer", "")).strip()
    answer_norm = normalize_text(answer)
    if not answer_norm:
        continue
    remove_options = []
    for removed_i in range(len(resolved)):
        retained = [resolved[i] for i in range(len(resolved)) if i != removed_i]
        retained_text = " ".join(f"{x[0]} {x[2]}" for x in retained)
        answer_absent = answer_norm in {"yes", "no"} or answer_norm not in normalize_text(retained_text)
        answer_tokens = set(answer_norm.split())
        retained_tokens = set(normalize_text(retained_text).split())
        if len(answer_tokens) > 1 and len(answer_tokens & retained_tokens) >= 2:
            answer_absent = False
        if answer_absent:
            remove_options.append((removed_i, retained))
    if not remove_options:
        continue
    removed_i, retained = remove_options[0]
    removed_title, removed_sid, removed_sentence = resolved[removed_i]
    support_keys = {(t, sid) for t, sid, _ in resolved}
    cands = []
    for title, sents in docs:
        for sid, sentence in enumerate(sents):
            sentence = sentence.strip()
            if not sentence or (title, sid) in support_keys:
                continue
            if answer_norm not in {"yes", "no"} and answer_norm in normalize_text(sentence):
                continue
            same_article = title == removed_title
            cands.append((not same_article, abs(len(sentence) - len(removed_sentence)), title, sid, sentence))
    if not cands:
        continue
    cands.sort()
    _, _, d_title, _, d_sent = cands[0]
    pos_passages = [f"[{t}] {s}" for t, _, s in resolved]
    neg_passages = [f"[{t}] {s}" for t, _, s in retained] + [f"[{d_title}] {d_sent}"]
    q = str(ex.get("question", "")).strip()
    pos_inp, pos_trunc = serialize_input(q, pos_passages)
    neg_inp, neg_trunc = serialize_input(q, neg_passages)
    if pos_inp == neg_inp:
        continue
    neg_norm = normalize_text(" ".join(neg_passages))
    ans_toks = set(answer_norm.split())
    if answer_norm not in {"yes", "no"} and answer_norm in neg_norm:
        continue
    if len(ans_toks) > 1 and len(ans_toks & set(neg_norm.split())) >= 2:
        continue
    pair_id = f"hotpotqa-{group_id}"
    b4_rows.append({
        "example_id": f"{pair_id}-positive", "pair_id": pair_id,
        "question": q, "input_text": pos_inp, "char_truncated": pos_trunc, "label": 1,
        "label_source": "HotpotQA distractor supporting_facts pair",
        "construction_method": "All official supporting facts supplied."
    })
    b4_rows.append({
        "example_id": f"{pair_id}-negative", "pair_id": pair_id,
        "question": q, "input_text": neg_inp, "char_truncated": neg_trunc, "label": 0,
        "label_source": "HotpotQA distractor supporting_facts pair",
        "construction_method": "One supporting fact replaced by nearest-length non-support sentence without answer leakage."
    })
benchmarks["04_hotpotqa_pairs"] = b4_rows
benchmark_sources["04_hotpotqa_pairs"] = {
    "url": b4_url, "sha256": sha256_file(b4_path), "n_examples": len(b4_rows), "n_pairs": len(b4_rows) // 2,
    "caveat": "Support-fact-derived proxy pairs from complete HotpotQA distractor validation split; full-question format."
}

# ----- Track 05: 2WikiMultiHopQA support-removal pairs (exact parity with v5) -----
def render_docs(docs):
    return "\n".join(f"[{title}] {sentence}" for title, sents in docs for sentence in sents)

b5_url = "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/dev.parquet?download=true"
b5_path = download_public_file(b5_url, CACHE_ROOT / "2wiki_dev.parquet")
tw_df = pd.read_parquet(b5_path)
tw_records = tw_df.to_dict("records")
tw_order = list(range(len(tw_records)))
random.Random(SEED).shuffle(tw_order)
b5_rows = []
for idx in tw_order:
    ex = tw_records[idx]
    question = str(ex.get("question", "")).strip()
    docs = context_documents(ex.get("context"))
    supports = support_tuples(ex.get("supporting_facts"))
    if not question or not docs or not supports:
        continue
    pos_passage = render_docs(docs)
    pos_inp, pos_trunc = serialize_input(question, pos_passage)
    answer = str(ex.get("answer", "")).strip()
    chosen_neg = None
    chosen_neg_inp = None
    chosen_neg_trunc = False
    for title, sent_idx in supports:
        neg_docs = []
        removed = False
        for doc_title, sents in docs:
            new_sents = list(sents)
            if not removed and doc_title == title and 0 <= sent_idx < len(new_sents):
                new_sents.pop(sent_idx)
                removed = True
            neg_docs.append((doc_title, new_sents))
        if not removed:
            continue
        neg_passage = render_docs(neg_docs)
        if answer and answer.casefold() in neg_passage.casefold():
            continue
        neg_inp, neg_trunc = serialize_input(question, neg_passage)
        if pos_inp == neg_inp:
            continue
        chosen_neg = neg_passage
        chosen_neg_inp = neg_inp
        chosen_neg_trunc = neg_trunc
        break
    if chosen_neg is None:
        continue
    pid = str(ex.get("_id", ex.get("id", idx)))
    pair_id = f"2wiki-{pid}"
    b5_rows.append({
        "example_id": f"{pair_id}-positive", "pair_id": pair_id,
        "question": question, "input_text": pos_inp, "char_truncated": pos_trunc, "label": 1,
        "label_source": "2Wiki support-fact-derived proxy label",
        "construction_method": "Full supplied context."
    })
    b5_rows.append({
        "example_id": f"{pair_id}-negative", "pair_id": pair_id,
        "question": question, "input_text": chosen_neg_inp, "char_truncated": chosen_neg_trunc, "label": 0,
        "label_source": "2Wiki support-fact-derived proxy label",
        "construction_method": "Deleted one supporting sentence without literal answer leakage."
    })
benchmarks["05_2wiki_support_pairs"] = b5_rows
benchmark_sources["05_2wiki_support_pairs"] = {
    "url": b5_url, "sha256": sha256_file(b5_path), "n_examples": len(b5_rows), "n_pairs": len(b5_rows) // 2,
    "caveat": "Support-deletion proxy pairs from full 2WikiMultiHopQA dev."
}

# ----- Track 06: FaithEval Unanswerable pairs (ICLR 2025, exact parity with v5) -----
b6_url = "https://huggingface.co/datasets/alvinming/FaithEval-unanswerable-v1.0-w-original_context/resolve/main/data/train-00000-of-00001.parquet?download=true"
b6_path = download_public_file(b6_url, CACHE_ROOT / "faitheval_unanswerable.parquet")
fe_df = pd.read_parquet(b6_path)
b6_rows = []
for idx, ex in enumerate(fe_df.to_dict("records")):
    qid = str(ex.get("qid", idx))
    question = str(ex.get("question", "")).strip()
    original = str(ex.get("original_context", "")).strip()
    edited = str(ex.get("context", "")).strip()
    if not question or not original or not edited:
        continue
    pair_id = f"faitheval-{qid}-row-{idx}"
    for suffix, passage, label in (("answerable", original, 1), ("unanswerable", edited, 0)):
        inp, trunc = serialize_input(question, passage)
        b6_rows.append({
            "example_id": f"{pair_id}-{suffix}", "pair_id": pair_id,
            "question": question, "input_text": inp, "char_truncated": trunc, "label": label,
            "label_source": "FaithEval original_context vs edited-unanswerable contrast",
            "construction_method": "Original answerable context (1) vs edited unanswerable context (0)."
        })
benchmarks["06_faitheval_pairs"] = b6_rows
benchmark_sources["06_faitheval_pairs"] = {
    "url": b6_url, "sha256": sha256_file(b6_path), "n_examples": len(b6_rows), "n_pairs": len(b6_rows) // 2,
    "caveat": "Community mirror with original_context paired against FaithEval unanswerable-v1.0."
}

# ----- Track 07: MultiHop-RAG (COLM 2024 — News Domain, native null_query vs answerable) -----
b7_q_url = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json?download=true"
b7_c_url = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/corpus.json?download=true"
b7_q_path = download_public_file(b7_q_url, CACHE_ROOT / "multihop_rag_queries.json")
b7_c_path = download_public_file(b7_c_url, CACHE_ROOT / "multihop_rag_corpus.json")
mh_queries = json.loads(b7_q_path.read_text(encoding="utf-8"))
mh_corpus = json.loads(b7_c_path.read_text(encoding="utf-8"))
corpus_snippets = [
    f"[{str(art.get('title', '')).strip()}] {str(art.get('body', '')).strip()[:500]}"
    for art in mh_corpus
]
mh_bm25 = SimpleBM25(corpus_snippets)
b7_rows = []
for idx, ex in enumerate(mh_queries):
    q_text = str(ex.get("query", "")).strip()
    q_type = str(ex.get("question_type", "")).strip()
    ev_list = ex.get("evidence_list", [])
    if not q_text:
        continue
    if q_type == "null_query" or not ev_list:
        label = 0
        # Retrieve top-3 realistic news snippets via BM25 so negatives have matched non-empty news context
        top_docs = mh_bm25.top_k(q_text, k=3)
        passages = [corpus_snippets[i] for i in top_docs]
        method = "Native null_query paired with top-3 BM25 news snippets from official MultiHop-RAG corpus."
    else:
        label = 1
        passages = [
            f"[{str(ev.get('title', '')).strip()}] {str(ev.get('fact', '')).strip()}"
            for ev in ev_list if str(ev.get("fact", "")).strip()
        ]
        if not passages:
            continue
        method = f"Native {q_type} paired with annotated multi-document evidence_list facts."
    inp, trunc = serialize_input(q_text, passages)
    b7_rows.append({
        "example_id": f"multihop-rag-{idx}", "pair_id": None,
        "question": q_text, "input_text": inp, "char_truncated": trunc, "label": label,
        "label_source": f"Official MultiHop-RAG question_type ({q_type})",
        "construction_method": method
    })
benchmarks["07_multihop_rag_news"] = b7_rows
benchmark_sources["07_multihop_rag_news"] = {
    "url": b7_q_url, "corpus_url": b7_c_url, "sha256": sha256_file(b7_q_path),
    "n_examples": len(b7_rows),
    "caveat": "Cross-domain news multi-hop benchmark (COLM 2024); imbalanced (301 null_query vs 2,255 answerable); negatives use top-3 BM25 news snippets."
}

# ----- Track 08: SQuAD 2.0 validation (11,873 examples, single-hop unanswerable control) -----
b8_url = "https://huggingface.co/datasets/rajpurkar/squad_v2/resolve/main/squad_v2/validation-00000-of-00001.parquet?download=true"
b8_path = download_public_file(b8_url, CACHE_ROOT / "squad_v2_val.parquet")
sq_df = pd.read_parquet(b8_path)
b8_rows = []
for idx, ex in enumerate(sq_df.to_dict("records")):
    q_text = str(ex.get("question", "")).strip()
    title = str(ex.get("title", "")).strip()
    ctx = str(ex.get("context", "")).strip()
    ans = ex.get("answers", {})
    ans_texts = as_list(ans.get("text")) if isinstance(ans, dict) else []
    label = 1 if len(ans_texts) > 0 else 0
    passage = f"[{title}] {ctx}" if title else ctx
    inp, trunc = serialize_input(q_text, passage)
    b8_rows.append({
        "example_id": f"squad2-{ex.get('id', idx)}", "pair_id": None,
        "question": q_text, "input_text": inp, "char_truncated": trunc, "label": label,
        "label_source": "Official SQuAD 2.0 validation answerability label",
        "construction_method": "Single-hop Wikipedia paragraph; label=1 iff answers.text is non-empty."
    })
benchmarks["08_squad_v2_dev"] = b8_rows
benchmark_sources["08_squad_v2_dev"] = {
    "url": b8_url, "sha256": sha256_file(b8_path), "n_examples": len(b8_rows),
    "caveat": "Single-hop answerability control (50/50 split) for comparison against DeBERTa-v3-large SQuAD2 SOTA."
}

# Print summary & compute signatures
total_examples = sum(len(v) for v in benchmarks.values())
print(f"\nPrepared {len(benchmarks)} tracks totaling {total_examples:,} examples:")
benchmark_signatures = {}
for name, rows in benchmarks.items():
    if not rows:
        raise ValueError(f"Prepared benchmark {name} has no examples.")
    vc = pd.Series([r["label"] for r in rows]).value_counts().to_dict()
    trunc_rate = float(np.mean([bool(r.get("char_truncated", False)) for r in rows]))
    print(f"  - {name}: {len(rows):,} rows | labels={vc} | char_trunc_rate={trunc_rate:.3f}")
    log_event("benchmark_prepared", benchmark=name, n_examples=len(rows), labels=vc, char_trunc_rate=round(trunc_rate, 4))
    h = hashlib.sha256()
    h.update(checkpoint_sha256.encode())
    h.update(json.dumps({"max_length": MAX_LENGTH, "passage_max_chars": PASSAGE_MAX_CHARS,
                         "threshold": DECISION_THRESHOLD, "label_mapping": {0: "NOT-ANSWERABLE", 1: "ANSWERABLE"}},
                        sort_keys=True).encode())
    h.update(rows_signature(rows).encode())
    benchmark_signatures[name] = h.hexdigest()

manifest = {
    "paper": "Diagnosing the Fact-Grounding Gap in Multi-Hop Question Answering (EMNLP 2026)",
    "paper_url": "https://aclanthology.org/2026.emnlp-main.834/",
    "checkpoint_file": str(checkpoint_file),
    "checkpoint_sha256": checkpoint_sha256,
    "checkpoint_bytes": checkpoint_file.stat().st_size,
    "matches_expected_local_sha256": checkpoint_sha256.upper() == EXPECTED_LOCAL_SHA256,
    "hardware": {"device": str(device), "n_gpus": n_gpus, "gpu_names": gpu_names},
    "base_protocol": {
        "max_length": MAX_LENGTH,
        "passage_max_chars": PASSAGE_MAX_CHARS,
        "threshold": DECISION_THRESHOLD,
        "effective_batch_size": effective_batch_size,
        "label_mapping": {"0": "NOT-ANSWERABLE", "1": "ANSWERABLE"},
    },
    "time_budget_hours": MAX_RUNTIME_HOURS,
    "total_examples": total_examples,
    "benchmark_sources": benchmark_sources,
    "benchmark_signatures": benchmark_signatures,
    "started_at_utc": utc_now(),
}
atomic_json(OUTPUT_ROOT / "run_manifest.json", manifest)

# ----- Metrics & Calibration Helpers -----
def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

def expected_calibration_error(y, p, bins=15):
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mask.any():
            total += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)

def calculate_metrics(records, expected_count, temperature=1.0, threshold=DECISION_THRESHOLD):
    y = np.asarray([int(r["label"]) for r in records], dtype=int)
    if temperature == 1.0:
        p = np.asarray([float(r["p_answerable"]) for r in records], dtype=float)
    else:
        diffs = np.asarray([float(r["logit_diff"]) for r in records], dtype=float)
        p = sigmoid(diffs / float(temperature))
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    out = {
        "n_expected": int(expected_count),
        "n_scored": int(len(records)),
        "complete": bool(len(records) == expected_count),
        "temperature": float(temperature),
        "threshold": float(threshold),
        "pos_rate_true": float(y.mean()) if len(y) else np.nan,
        "pos_rate_pred": float(pred.mean()) if len(y) else np.nan,
        "accuracy": float(accuracy_score(y, pred)) if len(y) else np.nan,
        "precision_answerable": float(precision_score(y, pred, zero_division=0)) if len(y) else np.nan,
        "recall_answerable": float(recall_score(y, pred, zero_division=0)) if len(y) else np.nan,
        "f1_answerable": float(f1_score(y, pred, zero_division=0)) if len(y) else np.nan,
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)) if len(y) else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(set(y)) > 1 else np.nan,
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        "mean_inference_ms": float(np.mean([r["inference_ms"] for r in records])) if records else np.nan,
        "p95_inference_ms": float(np.percentile([r["inference_ms"] for r in records], 95)) if records else np.nan,
        "char_trunc_rate": float(np.mean([bool(r.get("char_truncated", False)) for r in records])) if records else np.nan,
        "brier": float(brier_score_loss(y, p)) if len(y) else np.nan,
        "ece_15_bins": expected_calibration_error(y, p) if len(y) else np.nan,
        "auroc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else np.nan,
        "pr_auc_average_precision": float(average_precision_score(y, p)) if len(set(y)) > 1 else np.nan,
    }
    return out

def fit_temperature_and_threshold(records):
    y = np.asarray([int(r["label"]) for r in records], dtype=np.float64)
    diffs = np.asarray([float(r["logit_diff"]) for r in records], dtype=np.float64)
    def nll(temp):
        probs = np.clip(sigmoid(diffs / temp), 1e-7, 1.0 - 1e-7)
        return float(-np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))
    res = minimize_scalar(nll, bounds=(0.1, 20.0), method="bounded")
    best_t = float(res.x)
    cal_p = sigmoid(diffs / best_t)
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.20, 0.85, 66):
        f1 = f1_score(y.astype(int), (cal_p >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
    return best_t, best_thr, float(nll(1.0)), float(nll(best_t))

# ----- Durable, resumable inference loop with OOM backoff -----
metric_rows = []
timed_out = False
completed_names = []
current_batch_size = effective_batch_size

for name, rows in benchmarks.items():
    signature = benchmark_signatures[name]
    pred_path = PREDICTION_ROOT / f"{name}.jsonl"
    prior = state["benchmarks"].get(name, {})
    if pred_path.exists() and prior.get("signature") != signature:
        stale = pred_path.with_name(pred_path.stem + ".stale-" + datetime.now().strftime("%Y%m%dT%H%M%S") + pred_path.suffix)
        os.replace(pred_path, stale)
        print(f"Signature changed; preserved old predictions at {stale.name}")
    existing = safe_read_jsonl(pred_path) if state["benchmarks"].get(name, {}).get("signature") == signature else []
    rows_by_id = {str(r["example_id"]): r for r in rows}
    done_ids = {str(r["example_id"]) for r in existing} & set(rows_by_id)
    remaining = [r for r in rows if str(r["example_id"]) not in done_ids]
    print(f"\n[{name}] {len(done_ids):,}/{len(rows):,} durable | {len(remaining):,} remaining")
    log_event("benchmark_started", benchmark=name, already_durable=len(done_ids), remaining=len(remaining), expected=len(rows))

    pos = 0
    n_batches = 0
    while pos < len(remaining):
        if not check_time_budget():
            timed_out = True
            log_event("runtime_budget_reached", benchmark=name, rows_done=len(done_ids), rows_expected=len(rows))
            break
        batch = remaining[pos:pos + current_batch_size]
        texts = [canonical_text(r) for r in batch]
        try:
            encoded = tokenizer(texts, truncation=True, max_length=MAX_LENGTH,
                                padding=True, pad_to_multiple_of=8, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(**encoded).logits
                else:
                    logits = model(**encoded).logits
                if device.type == "cuda":
                    torch.cuda.synchronize()
                logits_f = logits.float().detach().cpu().numpy()
                probs = torch.softmax(torch.tensor(logits_f), dim=-1)[:, 1].numpy()
                logit_diffs = logits_f[:, 1] - logits_f[:, 0]
            elapsed_ms = (time.perf_counter() - t0) * 1000.0 / max(len(batch), 1)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and current_batch_size > 2:
                torch.cuda.empty_cache()
                current_batch_size = max(2, current_batch_size // 2)
                log_event("oom_backoff", benchmark=name, new_batch_size=current_batch_size)
                continue
            raise

        with open(pred_path, "a", encoding="utf-8") as f:
            for row, prob, ldiff in zip(batch, probs, logit_diffs):
                rec = {
                    "dataset": name,
                    "example_id": str(row["example_id"]),
                    "pair_id": row.get("pair_id"),
                    "question": str(row.get("question", "")),
                    "label": int(row["label"]),
                    "label_source": row.get("label_source", ""),
                    "char_truncated": bool(row.get("char_truncated", False)),
                    "logit_diff": float(ldiff),
                    "p_answerable": float(prob),
                    "prediction": int(float(prob) >= DECISION_THRESHOLD),
                    "threshold": DECISION_THRESHOLD,
                    "inference_ms": float(elapsed_ms),
                    "input_preview": canonical_text(row)[:600],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush(); os.fsync(f.fileno())

        done_ids.update(str(r["example_id"]) for r in batch)
        pos += len(batch)
        n_batches += 1
        state["benchmarks"][name] = {
            "signature": signature, "rows_done": len(done_ids), "rows_expected": len(rows),
            "last_batch_at_utc": utc_now(), "prediction_file": str(pred_path)
        }
        state["last_updated_utc"] = utc_now()
        atomic_json(STATE_PATH, state)
        if n_batches % 25 == 0 or pos >= len(remaining):
            log_event("batch_durable", benchmark=name, rows_done=len(done_ids), rows_expected=len(rows),
                      batch_size=len(batch), ms_per_ex=round(float(elapsed_ms), 2), elapsed_h=round(elapsed_hours(), 4))
            write_resume_bundle()

    current_records = [r for r in safe_read_jsonl(pred_path) if str(r.get("example_id")) in rows_by_id]
    if current_records:
        metric = {"benchmark": name, **calculate_metrics(current_records, len(rows))}
        metric_rows.append(metric)
        pd.DataFrame(metric_rows).to_csv(OUTPUT_ROOT / "metrics.csv", index=False)
    complete = len(done_ids) == len(rows)
    state["benchmarks"][name] = {
        "signature": signature, "rows_done": len(done_ids), "rows_expected": len(rows),
        "complete": complete, "last_updated_utc": utc_now(), "prediction_file": str(pred_path)
    }
    state["last_updated_utc"] = utc_now()
    atomic_json(STATE_PATH, state)
    log_event("benchmark_finished", benchmark=name, complete=complete, rows_done=len(done_ids), rows_expected=len(rows))
    write_resume_bundle()
    if complete:
        completed_names.append(name)
        print(f"  -> Completed {name}: F1={metric['f1_answerable']:.4f} | Acc={metric['accuracy']:.4f} | AUROC={metric['auroc']:.4f} | ECE={metric['ece_15_bins']:.4f}")
    if timed_out:
        break

# ----- Final aggregation: raw metrics, post-hoc calibrated metrics, pairwise ranking, errors -----
all_records_by_track = {}
for name, rows in benchmarks.items():
    p = PREDICTION_ROOT / f"{name}.jsonl"
    valid_ids = {str(r["example_id"]) for r in rows}
    recs = [r for r in safe_read_jsonl(p) if str(r.get("example_id")) in valid_ids]
    # Deduplicate by example_id preserving latest
    dedup = {str(r["example_id"]): r for r in recs}
    all_records_by_track[name] = list(dedup.values())

raw_metrics = []
for name, rows in benchmarks.items():
    recs = all_records_by_track.get(name, [])
    if recs:
        raw_metrics.append({"benchmark": name, **calculate_metrics(recs, len(rows))})
metrics_df = pd.DataFrame(raw_metrics)
metrics_df.to_csv(OUTPUT_ROOT / "metrics.csv", index=False)

# Fit post-hoc temperature scaling & threshold strictly on 01_musique_dev (zero OOD leakage)
cal_info = {}
cal_metrics = []
if all_records_by_track.get("01_musique_dev"):
    best_t, best_thr, nll_before, nll_after = fit_temperature_and_threshold(all_records_by_track["01_musique_dev"])
    cal_info = {
        "fit_on": "01_musique_dev",
        "temperature": best_t,
        "f1_optimal_threshold_on_musique_dev": best_thr,
        "musique_nll_before": nll_before,
        "musique_nll_after": nll_after,
    }
    print(f"\nPost-hoc calibration (fit strictly on 01_musique_dev): T={best_t:.4f}, thr={best_thr:.3f}, NLL {nll_before:.4f} -> {nll_after:.4f}")
    for name, rows in benchmarks.items():
        recs = all_records_by_track.get(name, [])
        if recs:
            cal_metrics.append({"benchmark": name, **calculate_metrics(recs, len(rows), temperature=best_t, threshold=0.5)})
            cal_metrics.append({"benchmark": f"{name}__cal_thr", **calculate_metrics(recs, len(rows), temperature=best_t, threshold=best_thr)})
    pd.DataFrame(cal_metrics).to_csv(OUTPUT_ROOT / "metrics_calibrated.csv", index=False)

# Pairwise ranking diagnostic on matched pair benchmarks
pair_metrics = []
all_pair_scores = []
for name in ("04_hotpotqa_pairs", "05_2wiki_support_pairs", "06_faitheval_pairs"):
    recs = all_records_by_track.get(name, [])
    if not recs:
        continue
    df_p = pd.DataFrame(recs)
    paired = df_p[df_p["pair_id"].notna()]
    scores = []
    for pid, grp in paired.groupby("pair_id"):
        pos = grp[grp["label"] == 1]
        neg = grp[grp["label"] == 0]
        if len(pos) == 1 and len(neg) == 1:
            delta = float(pos.iloc[0]["p_answerable"] - neg.iloc[0]["p_answerable"])
            ldelta = float(pos.iloc[0]["logit_diff"] - neg.iloc[0]["logit_diff"])
            corr = 1.0 if ldelta > 0 else (0.5 if ldelta == 0 else 0.0)
            scores.append((corr, delta, ldelta))
            all_pair_scores.append((corr, delta, ldelta))
    if scores:
        pair_metrics.append({
            "benchmark": name, "n_pairs": len(scores),
            "pairwise_accuracy_ties_half": float(np.mean([x[0] for x in scores])),
            "mean_prob_margin_pos_minus_neg": float(np.mean([x[1] for x in scores])),
            "mean_logit_margin_pos_minus_neg": float(np.mean([x[2] for x in scores])),
        })
if all_pair_scores:
    pair_metrics.append({
        "benchmark": "ALL_MATCHED_PAIRS", "n_pairs": len(all_pair_scores),
        "pairwise_accuracy_ties_half": float(np.mean([x[0] for x in all_pair_scores])),
        "mean_prob_margin_pos_minus_neg": float(np.mean([x[1] for x in all_pair_scores])),
        "mean_logit_margin_pos_minus_neg": float(np.mean([x[2] for x in all_pair_scores])),
    })
pairwise_df = pd.DataFrame(pair_metrics)
pairwise_df.to_csv(OUTPUT_ROOT / "pairwise_ranking_metrics.csv", index=False)

# Save compact top-100 highest-confidence FP and FN per track for qualitative error inspection
err_samples = []
for name, recs in all_records_by_track.items():
    fps = sorted([r for r in recs if r["label"] == 0 and r["prediction"] == 1], key=lambda x: -x["p_answerable"])[:50]
    fns = sorted([r for r in recs if r["label"] == 1 and r["prediction"] == 0], key=lambda x: x["p_answerable"])[:50]
    for r in fps:
        err_samples.append({**r, "error_type": "FP"})
    for r in fns:
        err_samples.append({**r, "error_type": "FN"})
if err_samples:
    pd.DataFrame(err_samples).to_csv(OUTPUT_ROOT / "error_cases_top100_per_track.csv", index=False)

peak_vram_gib = [
    round(torch.cuda.max_memory_allocated(i) / (1024 ** 3), 3) for i in range(n_gpus)
] if n_gpus else []

manifest["finished_at_utc"] = utc_now()
manifest["elapsed_hours"] = round(elapsed_hours(), 4)
manifest["timed_out_at_budget"] = bool(timed_out)
manifest["completed_benchmarks"] = completed_names
manifest["calibration_fit_on_musique_dev"] = cal_info
manifest["peak_vram_gib_per_gpu"] = peak_vram_gib
manifest["progress_state"] = state.get("benchmarks", {})
atomic_json(OUTPUT_ROOT / "run_manifest.json", manifest)
atomic_json(STATE_PATH, state)
write_resume_bundle()

print(f"\n=== ALL DONE | Elapsed: {elapsed_hours():.2f}h | Completed {len(completed_names)}/{len(benchmarks)} tracks ===")
print("Peak VRAM (GiB):", peak_vram_gib)
for fn in ["metrics.csv", "metrics_calibrated.csv", "pairwise_ranking_metrics.csv",
           "error_cases_top100_per_track.csv", "run_manifest.json", "progress_state.json", "resume_bundle.zip"]:
    fp = OUTPUT_ROOT / fn
    if fp.exists():
        print(f"  - {fn}: {fp.stat().st_size:,} bytes")
if not metrics_df.empty:
    display(metrics_df[["benchmark", "n_scored", "accuracy", "precision_answerable", "recall_answerable",
                        "f1_answerable", "macro_f1", "auroc", "ece_15_bins", "mean_inference_ms"]])
if not pairwise_df.empty:
    display(pairwise_df)
display(FileLink(str(OUTPUT_ROOT / "resume_bundle.zip")))
'''


def md_cell(source: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main():
    notebook = {
        "cells": [
            md_cell(INTRO_MD, "overview"),
            code_cell(SETUP_CELL, "setup"),
            code_cell(MAIN_CELL, "run_full_8_benchmark_eval"),
        ],
        "metadata": {
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "dataSources": [],
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    STAGING.mkdir(parents=True, exist_ok=True)
    nb_path = ROOT / "05_full_decision_benchmark_eval.ipynb"
    staged_path = STAGING / "05_full_decision_benchmark_eval.ipynb"
    for p in (nb_path, staged_path):
        p.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Wrote:", nb_path)
    print("Wrote:", staged_path)


if __name__ == "__main__":
    main()
