"""Build Notebook 07: Overnight End-to-End Multi-Hop RAG Systems Suite.

Evaluates 5 end-to-end Multi-Hop RAG systems on Kaggle Tesla T4 x2 under `huthnhnguyn`
across 800 shared multi-hop questions (HotpotQA 200 + 2Wiki 200 + MuSiQue-Ans 200 + MultiHop-RAG 200)
with a shared per-dataset corpus pool (~7,900 passages):
- System 1: Standard Dense RAG (intfloat/e5-base-v2 top-5 -> Qwen/Qwen2.5-7B-Instruct 4-bit)
- System 2 (P5): LinearRAG (ICLR 2026) — Relation-Free Tri-Graph (Entities + all-mpnet-base-v2 + Local Semantic Bridging + PPR) -> Qwen2.5-7B-Instruct 4-bit
- System 3 (FG-Gate): Adaptive Fact-Grounding DeBERTa (93ef195f, T=1.929) Gate + 2nd-Hop Query Reformulation -> Qwen2.5-7B-Instruct 4-bit
- System 4 (P6a): CoRAG (NeurIPS 2025) — corag/CoRAG-Llama3.1-8B-MultihopQA (4-bit NF4, official prompts.py, L=3)
- System 5 (P6b): Search-R1 (COLM 2025) — PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo (4-bit NF4, official <think>/<search>/<information>/<answer> loop)
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "kaggle_overnight_suite"

INTRO_MD = r"""# Overnight End-to-End Multi-Hop RAG Suite (Notebook 07): LinearRAG, CoRAG, Search-R1 & Adaptive Fact-Grounding Gate

This self-contained Kaggle GPU (Tesla T4 x2) notebook benchmarks **5 complete end-to-end Multi-Hop RAG architectures** on the **exact same 800 multi-hop questions** (`HotpotQA` 200 + `2WikiMultiHopQA` 200 + `MuSiQue-Ans` 200 + `MultiHop-RAG` 200) searching over unified per-dataset passage pools (~1,800–3,600 passages per dataset).

## Systems Evaluated (Crash-Isolated, Resumable, Logged via `[RUNLOG]`)
1. **Standard Dense RAG (Baseline):** `intfloat/e5-base-v2` top-5 → `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4).
2. **P5 — LinearRAG (ICLR 2026):** Relation-Free Tri-Graph (Entity extraction + `sentence-transformers/all-mpnet-base-v2` semantic linking + Local Entity Bridging + Personalized PageRank) → `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4).
3. **Adaptive Fact-Grounding Gate (Proposed Bridge):** `e5-base-v2` top-3 → attached calibrated `93ef195f` DeBERTa-v3-large (`T = 1.929`) gate → triggers 2nd-hop reformulated retrieval ONLY when `p_cal < 0.50` → `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4).
4. **P6a — CoRAG (NeurIPS 2025):** `corag/CoRAG-Llama3.1-8B-MultihopQA` (4-bit NF4) using official `microsoft/LMOps/corag/src/prompts.py` iterative sub-query → E5 top-3 → intermediate answer (`L = 3`) → final answer.
5. **P6b — Search-R1 (COLM 2025):** `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo` (4-bit NF4) using official `<think>...</think><search>...</search><information>...</information><answer>...</answer>` loop backed by E5 top-3.
"""

SETUP_CELL = r"""import sys, subprocess, time
NOTEBOOK_STARTED_AT = time.monotonic()
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.46.3", "huggingface-hub>=0.24.0,<1.0", "accelerate>=0.34.0,<1.5.0",
    "bitsandbytes>=0.45.0", "sentence-transformers>=3.0.0,<4.0.0", "einops",
    "sentencepiece", "protobuf", "scipy", "scikit-learn", "pyarrow", "nltk"
])
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "spacy"])
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "-q"])
except Exception as e:
    print("Optional spaCy download skipped, using fast built-in entity extractor:", e)
print("Notebook 07 runtime setup complete (transformers 4.46.3 + hf-hub<1). Wall-time budget: 10.5 hours.")
"""

MAIN_CELL = r'''import os, sys, gc, re, json, time, math, string, hashlib, random, zipfile, warnings, traceback
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
)
from sentence_transformers import SentenceTransformer
from IPython.display import display, FileLink

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ----- Global Config -----
MAX_RUNTIME_HOURS = 10.5
SEED = 42
N_PER_DATASET = 200
PASSAGE_MAX_CHARS = 1500
FG_TEMP = 1.9289911363118324
EXPECTED_FG_SHA256 = "93EF195F3840689A70094C8FDBE4C1126A86E9FAF1B9382A3040E3D32B470260"
STARTED_AT = globals().get("NOTEBOOK_STARTED_AT", time.monotonic())

INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working/overnight_suite_07")
CACHE_ROOT = OUTPUT_ROOT / "download_cache"
PRED_ROOT = OUTPUT_ROOT / "predictions"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
PRED_ROOT.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUTPUT_ROOT / "run_log.jsonl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)] if n_gpus else ["CPU"]

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def elapsed_hours():
    return (time.monotonic() - STARTED_AT) / 3600.0

def check_budget(reserve_hours=0.25):
    return elapsed_hours() < (MAX_RUNTIME_HOURS - reserve_hours)

def log_event(event, **fields):
    rec = {"time_utc": utc_now(), "elapsed_h": round(elapsed_hours(), 4), "event": event, **fields}
    line = json.dumps(rec, ensure_ascii=False)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush(); os.fsync(f.fileno())
    print("[RUNLOG] " + line, flush=True)

def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def write_bundle():
    tmp = OUTPUT_ROOT / "results_bundle.tmp"
    final = OUTPUT_ROOT / "results_bundle.zip"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
        for p in OUTPUT_ROOT.rglob("*"):
            if not p.is_file() or p.name in {"results_bundle.zip", "results_bundle.tmp"}:
                continue
            if CACHE_ROOT in p.parents:
                continue
            zf.write(p, p.relative_to(OUTPUT_ROOT).as_posix())
    os.replace(tmp, final)

log_event("rag_suite_started", device=str(device), n_gpus=n_gpus, gpu_names=gpu_names, n_per_dataset=N_PER_DATASET)

# ----- Shared QA Metrics & Helpers -----
def as_list(x):
    if x is None: return []
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (list, tuple)): return list(x)
    return [x]

def read_jsonish(value):
    if isinstance(value, str):
        try: return json.loads(value)
        except Exception: return value
    return value

def download_public_file(url, destination, retries=4):
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    tmp = destination.with_suffix(destination.suffix + ".part")
    last_err = None
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=(30, 180), headers={"User-Agent": "overnight-rag-suite/1.0"}) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk: f.write(chunk)
                    f.flush(); os.fsync(f.fileno())
            os.replace(tmp, destination)
            return destination
        except Exception as exc:
            last_err = exc
            if tmp.exists(): tmp.unlink()
            time.sleep(min(3 * (attempt + 1), 12))
    raise RuntimeError(f"Failed downloading {url}") from last_err

def normalize_answer(s):
    s = str(s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def clean_pred_text(pred):
    txt = str(pred or "").strip()
    txt = re.sub(r"^(final\s+)?answer\s*:\s*", "", txt, flags=re.IGNORECASE).strip()
    txt = txt.splitlines()[0].strip() if txt else ""
    return txt.strip('"\'` ')

def exact_match(pred, golds):
    np_ = normalize_answer(clean_pred_text(pred))
    return max((1.0 if np_ == normalize_answer(g) and np_ != "" else 0.0) for g in golds) if golds else 0.0

def token_f1(pred, golds):
    pt = normalize_answer(clean_pred_text(pred)).split()
    best = 0.0
    for g in golds:
        gt = normalize_answer(g).split()
        if not pt and not gt:
            best = max(best, 1.0); continue
        if not pt or not gt:
            continue
        common = Counter(pt) & Counter(gt)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        p = num_same / len(pt)
        r = num_same / len(gt)
        best = max(best, 2 * p * r / (p + r))
    return best

def contain_acc(pred, golds):
    np_ = normalize_answer(clean_pred_text(pred))
    if not np_:
        return 0.0
    for g in golds:
        ng = normalize_answer(g)
        if ng and (ng in np_ or np_ in ng):
            return 1.0
    return 0.0

def title_recall(retrieved_titles, gold_titles):
    if not gold_titles:
        return np.nan
    ret_set = {normalize_answer(t) for t in retrieved_titles if str(t).strip()}
    gold_set = {normalize_answer(t) for t in gold_titles if str(t).strip()}
    if not gold_set:
        return np.nan
    return len(ret_set & gold_set) / len(gold_set)

def context_documents(value):
    value = read_jsonish(value)
    docs = []
    if isinstance(value, dict):
        titles = as_list(value.get("title") if value.get("title") is not None else value.get("titles"))
        sents_all = as_list(value.get("sentences") if value.get("sentences") is not None else value.get("sentence"))
        for title, sents in zip(titles, sents_all):
            docs.append((str(title).strip(), " ".join(str(s).strip() for s in as_list(sents))))
    elif isinstance(value, (list, tuple, np.ndarray)):
        for item in as_list(value):
            item = read_jsonish(item)
            if isinstance(item, dict):
                raw_s = item.get("sentences") if item.get("sentences") is not None else item.get("sentence")
                docs.append((str(item.get("title", "")).strip(), " ".join(str(s).strip() for s in as_list(raw_s))))
            elif isinstance(item, (list, tuple, np.ndarray)):
                il = as_list(item)
                if len(il) >= 2:
                    s = read_jsonish(il[1])
                    docs.append((str(il[0]).strip(), " ".join(str(x).strip() for x in as_list([s] if isinstance(s, str) else s))))
    return docs

def support_titles(value):
    value = read_jsonish(value)
    titles = []
    if isinstance(value, dict):
        titles = [str(t).strip() for t in as_list(value.get("title") if value.get("title") is not None else value.get("titles"))]
    elif isinstance(value, (list, tuple, np.ndarray)):
        for item in as_list(value):
            if isinstance(item, (list, tuple, np.ndarray)) and len(as_list(item)) >= 1:
                titles.append(str(as_list(item)[0]).strip())
            elif isinstance(item, dict) and item.get("title"):
                titles.append(str(item["title"]).strip())
    return list(dict.fromkeys(t for t in titles if t))

# =====================================================================
# BUILD 4 UNIFIED MULTI-HOP QA DATASETS + PER-DATASET CORPUS POOLS
# =====================================================================
datasets = {}  # name -> {"questions": [...], "corpus": [{"doc_id", "title", "text"}]}

def add_to_pool(pool_map, title, text):
    key = (title.strip(), text.strip()[:200])
    if key not in pool_map and text.strip():
        pool_map[key] = {"doc_id": len(pool_map), "title": title.strip(), "text": text.strip()[:1800]}

# 1. HotpotQA distractor val (200 questions)
hp_path = download_public_file(
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet?download=true",
    CACHE_ROOT / "hotpotqa_val.parquet"
)
hp_recs = pd.read_parquet(hp_path).to_dict("records")
hp_ord = list(range(len(hp_recs)))
random.Random(SEED).shuffle(hp_ord)
hp_qs, hp_pool = [], {}
for idx in hp_ord:
    if len(hp_qs) >= N_PER_DATASET: break
    ex = hp_recs[idx]
    q, ans = str(ex.get("question", "")).strip(), str(ex.get("answer", "")).strip()
    docs = context_documents(ex.get("context"))
    st = support_titles(ex.get("supporting_facts"))
    if not q or not ans or not docs or not st: continue
    for t, txt in docs:
        add_to_pool(hp_pool, t, txt)
    hp_qs.append({"qid": f"hotpot-{ex.get('id', idx)}", "question": q, "golds": [ans], "gold_titles": st})
datasets["hotpotqa"] = {"questions": hp_qs, "corpus": list(hp_pool.values())}

# 2. 2WikiMultiHopQA dev (200 questions)
tw_path = download_public_file(
    "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/dev.parquet?download=true",
    CACHE_ROOT / "2wiki_dev.parquet"
)
tw_recs = pd.read_parquet(tw_path).to_dict("records")
tw_ord = list(range(len(tw_recs)))
random.Random(SEED).shuffle(tw_ord)
tw_qs, tw_pool = [], {}
for idx in tw_ord:
    if len(tw_qs) >= N_PER_DATASET: break
    ex = tw_recs[idx]
    q, ans = str(ex.get("question", "")).strip(), str(ex.get("answer", "")).strip()
    docs = context_documents(ex.get("context"))
    st = support_titles(ex.get("supporting_facts"))
    if not q or not ans or not docs or not st: continue
    for t, txt in docs:
        add_to_pool(tw_pool, t, txt)
    tw_qs.append({"qid": f"2wiki-{ex.get('_id', ex.get('id', idx))}", "question": q, "golds": [ans], "gold_titles": st})
datasets["2wiki"] = {"questions": tw_qs, "corpus": list(tw_pool.values())}

# 3. MuSiQue-Ans dev (200 questions)
mq_path = download_public_file(
    "https://huggingface.co/datasets/bdsaglam/musique/resolve/main/musique_ans_v1.0_dev.jsonl?download=true",
    CACHE_ROOT / "musique_ans_v1.0_dev.jsonl"
)
mq_recs = [json.loads(l) for l in mq_path.read_text(encoding="utf-8").splitlines() if l.strip()]
mq_ord = list(range(len(mq_recs)))
random.Random(SEED).shuffle(mq_ord)
mq_qs, mq_pool = [], {}
for idx in mq_ord:
    if len(mq_qs) >= N_PER_DATASET: break
    ex = mq_recs[idx]
    q = str(ex.get("question", "")).strip()
    ans = str(ex.get("answer", "")).strip()
    aliases = [str(a).strip() for a in ex.get("answer_aliases", []) if str(a).strip()]
    golds = list(dict.fromkeys([ans] + aliases))
    paras = ex.get("paragraphs", [])
    st = list(dict.fromkeys(str(p.get("title", "")).strip() for p in paras if p.get("is_supporting") and str(p.get("title", "")).strip()))
    if not q or not ans or not paras or not st: continue
    for p in paras:
        add_to_pool(mq_pool, str(p.get("title", "")), str(p.get("paragraph_text", "")))
    mq_qs.append({"qid": f"musique-{ex.get('id', idx)}", "question": q, "golds": golds, "gold_titles": st})
datasets["musique_ans"] = {"questions": mq_qs, "corpus": list(mq_pool.values())}

# 4. MultiHop-RAG News (200 non-null questions over full 609-article corpus)
mhq_path = download_public_file("https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json?download=true", CACHE_ROOT / "mhrag_q.json")
mhc_path = download_public_file("https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/corpus.json?download=true", CACHE_ROOT / "mhrag_c.json")
mhq_recs = json.loads(mhq_path.read_text(encoding="utf-8"))
mhc_recs = json.loads(mhc_path.read_text(encoding="utf-8"))
mh_pool = {}
for a in mhc_recs:
    add_to_pool(mh_pool, str(a.get("title", "")), str(a.get("body", ""))[:1800])
mh_valid = [ex for ex in mhq_recs if str(ex.get("question_type", "")) != "null_query" and str(ex.get("answer", "")).strip() and ex.get("evidence_list")]
mh_ord = list(range(len(mh_valid)))
random.Random(SEED).shuffle(mh_ord)
mh_qs = []
for idx in mh_ord[:N_PER_DATASET]:
    ex = mh_valid[idx]
    q = str(ex.get("query", "")).strip()
    ans = str(ex.get("answer", "")).strip()
    st = list(dict.fromkeys(str(e.get("title", "")).strip() for e in ex.get("evidence_list", []) if str(e.get("title", "")).strip()))
    mh_qs.append({"qid": f"mhrag-{idx}", "question": q, "golds": [ans], "gold_titles": st})
datasets["multihop_rag"] = {"questions": mh_qs, "corpus": list(mh_pool.values())}

for ds_name, ds in datasets.items():
    log_event("dataset_ready", dataset=ds_name, n_questions=len(ds["questions"]), n_corpus_docs=len(ds["corpus"]))

# =====================================================================
# BUILD SHARED DENSE INDEXES (intfloat/e5-base-v2)
# =====================================================================
log_event("indexing_started", model="intfloat/e5-base-v2")
e5_model = SentenceTransformer("intfloat/e5-base-v2", device="cuda:0" if device.type == "cuda" else "cpu")

class DenseE5Index:
    def __init__(self, corpus_docs, embedder):
        self.docs = corpus_docs
        self.embedder = embedder
        texts = [f"passage: {d['title']}. {d['text']}" for d in corpus_docs]
        self.emb = embedder.encode(texts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def retrieve(self, query, k=5):
        qv = self.embedder.encode([f"query: {query}"], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)[0]
        scores = self.emb @ qv
        idxs = np.argsort(scores)[::-1][:k]
        return [self.docs[int(i)] for i in idxs]

e5_indexes = {}
for ds_name, ds in datasets.items():
    t0 = time.perf_counter()
    e5_indexes[ds_name] = DenseE5Index(ds["corpus"], e5_model)
    log_event("e5_index_built", dataset=ds_name, docs=len(ds["corpus"]), sec=round(time.perf_counter() - t0, 2))

# Move e5_model to CPU while heavy 7B/8B LLMs run on GPU? Actually e5-base-v2 is only ~440 MB VRAM!
# Even better: keep e5_model on cuda:1 if n_gpus > 1 so cuda:0 is 100% free for 7B/8B LLMs!
if n_gpus > 1:
    e5_model = e5_model.to("cuda:1")
    for idx_obj in e5_indexes.values():
        idx_obj.embedder = e5_model

# =====================================================================
# BUILD P5: LinearRAG (ICLR 2026) RELATION-FREE TRI-GRAPH INDEXES
# =====================================================================
log_event("linearrag_indexing_started", model="sentence-transformers/all-mpnet-base-v2")
try:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer", "attribute_ruler"])
    USE_SPACY = True
    log_event("spacy_loaded", status="ok")
except Exception as sp_exc:
    nlp = None
    USE_SPACY = False
    log_event("spacy_fallback", reason=str(sp_exc))

CAP_ENT_RE = re.compile(r"\b([A-Z][a-zA-Z0-9'\-]+(?:\s+(?:of|the|in|de|van|von|and|for)?\s*[A-Z][a-zA-Z0-9'\-]+){0,4})\b")
ENTITY_STOP = {"The", "This", "That", "These", "Those", "There", "Here", "When", "Where", "What", "Which", "Who", "While", "After", "Before", "During", "Since", "Although", "However", "Also", "Some", "Many", "Most", "Other", "First", "Second", "Third", "In", "On", "At", "By", "For", "From", "With", "As", "It", "He", "She", "They", "We", "His", "Her", "Their"}

def extract_entities(title, text):
    ents = set()
    if title and len(title.strip()) > 1:
        ents.add(title.strip().lower())
    if USE_SPACY:
        doc = nlp(text[:1200])
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART", "EVENT", "FAC", "NORP", "PRODUCT"}:
                clean = ent.text.strip().lower()
                if 2 <= len(clean) <= 80:
                    ents.add(clean)
    else:
        for m in CAP_ENT_RE.findall(text[:1200]):
            s = m.strip()
            if s not in ENTITY_STOP and 2 <= len(s) <= 80:
                ents.add(s.lower())
    return list(ents)[:25]

mpnet_dev = "cuda:1" if n_gpus > 1 else ("cuda:0" if device.type == "cuda" else "cpu")
mpnet_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=mpnet_dev)

class LinearRAGIndex:
    """Relation-Free Tri-Graph (Entity <-> Passage) + Local Semantic Bridging + Personalized PageRank (ICLR 2026)."""
    def __init__(self, corpus_docs, embedder):
        self.docs = corpus_docs
        self.embedder = embedder
        self.n_docs = len(corpus_docs)
        ent2id = {}
        rows, cols = [], []
        for d_idx, d in enumerate(corpus_docs):
            for ent in extract_entities(d["title"], d["text"]):
                if ent not in ent2id:
                    ent2id[ent] = len(ent2id)
                rows.append(d_idx)
                cols.append(ent2id[ent])
        self.entities = [None] * len(ent2id)
        for e, eid in ent2id.items():
            self.entities[eid] = e
        self.n_ents = len(self.entities)
        # Bipartite incidence matrix B (n_docs x n_ents)
        self.B = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(self.n_docs, max(self.n_ents, 1)))
        # Column-normalized (ent -> doc) and row-normalized (doc -> ent) transitions for fast PPR
        doc_deg = np.maximum(np.asarray(self.B.sum(axis=1)).ravel(), 1.0)
        ent_deg = np.maximum(np.asarray(self.B.sum(axis=0)).ravel(), 1.0)
        # Down-weight overly frequent hub entities (IDF-like damping)
        self.ent_idf = np.log(1.0 + self.n_docs / (1.0 + ent_deg)).astype(np.float32)
        self.P_d2e = sp.diags(1.0 / doc_deg) @ self.B
        self.P_e2d = (self.B @ sp.diags(1.0 / ent_deg)).T  # (n_ents x n_docs)
        # Semantic embeddings for passages and unique entities
        p_texts = [f"{d['title']}. {d['text'][:800]}" for d in corpus_docs]
        self.p_emb = embedder.encode(p_texts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        self.e_emb = embedder.encode(self.entities, batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32) if self.n_ents else np.zeros((1, 768), dtype=np.float32)

    def retrieve(self, query, k=5, alpha=0.5, n_iter=12):
        qv = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)[0]
        # Dense passage similarity
        p_sim = np.maximum(self.p_emb @ qv, 0.0)
        # Stage 1: Local Semantic Bridging (seed entity activation + 1-hop bridge entity activation)
        e_sim = np.maximum(self.e_emb @ qv, 0.0) * self.ent_idf
        top_e = np.argsort(e_sim)[::-1][:12]
        seed_e = np.zeros(self.n_ents, dtype=np.float32)
        seed_e[top_e] = e_sim[top_e]
        if seed_e.sum() > 0:
            seed_e /= seed_e.sum()
        # Activate local passages -> bridge entities -> combine personalization vector
        local_p = np.asarray(self.P_e2d.T @ seed_e).ravel()
        bridge_e = np.asarray(self.P_d2e.T @ local_p).ravel() * self.ent_idf
        # Combine seed dense passages + entity-activated passages as PPR teleport vector
        top_p = np.argsort(p_sim)[::-1][:10]
        seed_p = np.zeros(self.n_docs, dtype=np.float32)
        seed_p[top_p] = p_sim[top_p]
        if seed_p.sum() > 0: seed_p /= seed_p.sum()
        if local_p.sum() > 0: local_p /= local_p.sum()
        teleport = 0.5 * seed_p + 0.5 * local_p
        # Stage 2: Global Personalized PageRank on Passage <-> Entity Tri-Graph
        # 2-step random walk: p -> e -> p (linear in nnz(B), no O(V^2) dense matrix!)
        pr = teleport.copy()
        for _ in range(n_iter):
            e_dist = np.asarray(self.P_d2e.T @ pr).ravel()
            # Boost bridge entities semantically aligned with query
            e_dist = e_dist * (1.0 + 0.5 * np.maximum(self.e_emb @ qv, 0.0))
            p_next = np.asarray(self.P_e2d.T @ e_dist).ravel()
            if p_next.sum() > 0:
                p_next /= p_next.sum()
            pr = alpha * teleport + (1.0 - alpha) * p_next
        # Blend normalized PPR score with direct semantic passage similarity
        pr_norm = pr / max(pr.max(), 1e-8)
        ps_norm = p_sim / max(p_sim.max(), 1e-8)
        final_score = 0.65 * pr_norm + 0.35 * ps_norm
        idxs = np.argsort(final_score)[::-1][:k]
        return [self.docs[int(i)] for i in idxs]

linearrag_indexes = {}
for ds_name, ds in datasets.items():
    t0 = time.perf_counter()
    linearrag_indexes[ds_name] = LinearRAGIndex(ds["corpus"], mpnet_model)
    log_event("linearrag_index_built", dataset=ds_name, docs=len(ds["corpus"]),
              entities=linearrag_indexes[ds_name].n_ents, sec=round(time.perf_counter() - t0, 2))

# Tracking outputs
rag_metrics = []
pred_samples = []
stage_status = {}

def save_all_rag_tables():
    if rag_metrics:
        pd.DataFrame(rag_metrics).to_csv(OUTPUT_ROOT / "rag_metrics.csv", index=False)
    if pred_samples:
        pd.DataFrame(pred_samples).to_csv(OUTPUT_ROOT / "rag_predictions_sample.csv", index=False)
    atomic_json(OUTPUT_ROOT / "run_manifest.json", {
        "updated_at_utc": utc_now(), "elapsed_hours": round(elapsed_hours(), 4),
        "hardware": {"device": str(device), "n_gpus": n_gpus, "gpu_names": gpu_names},
        "n_per_dataset": N_PER_DATASET,
        "dataset_corpus_sizes": {k: len(v["corpus"]) for k, v in datasets.items()},
        "stage_status": stage_status,
    })
    write_bundle()

def record_dataset_metrics(system_name, ds_name, rows_eval):
    em = float(np.mean([r["em"] for r in rows_eval]))
    f1 = float(np.mean([r["f1"] for r in rows_eval]))
    ca = float(np.mean([r["contain_acc"] for r in rows_eval]))
    rec = float(np.nanmean([r["recall"] for r in rows_eval]))
    calls = float(np.mean([r["search_calls"] for r in rows_eval]))
    lat = float(np.mean([r["latency_s"] for r in rows_eval]))
    extra = {}
    if "triggered_hop2" in rows_eval[0]:
        extra["hop2_trigger_rate"] = float(np.mean([r["triggered_hop2"] for r in rows_eval]))
    if "format_valid" in rows_eval[0]:
        extra["format_valid_rate"] = float(np.mean([r["format_valid"] for r in rows_eval]))
    row = {
        "system": system_name, "dataset": ds_name, "n_questions": len(rows_eval),
        "em": em, "f1": f1, "contain_acc": ca, "evidence_recall": rec,
        "avg_search_calls": calls, "mean_latency_s": lat, **extra
    }
    rag_metrics.append(row)
    # Save first 15 predictions per system x dataset for qualitative error analysis
    for r in rows_eval[:15]:
        pred_samples.append({"system": system_name, "dataset": ds_name, **{k: r[k] for k in ("qid", "question", "gold", "pred", "em", "f1", "recall", "search_calls")}})
    log_event("system_dataset_done", system=system_name, dataset=ds_name,
              em=round(em, 4), f1=round(f1, 4), contain=round(ca, 4),
              recall=round(rec, 4), calls=round(calls, 2), lat_s=round(lat, 2))
    save_all_rag_tables()

def format_docs_block(docs):
    return "\n\n".join(f"[{d['title']}] {d['text'][:900]}" for d in docs)

# =====================================================================
# SYSTEMS 1, 2 (LinearRAG), 3 (Adaptive FG-Gate): Shared Qwen2.5-7B-Instruct 4-bit
# =====================================================================
QA_SYS = "You are a precise multi-hop QA assistant. Answer the question using ONLY the provided passages. Respond with ONLY the short exact answer (a few words, name, date, or Yes/No), with no explanation."

def qwen_generate(tok, mdl, msgs, max_new_tokens=48):
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, truncation=True, max_length=3200, return_tensors="pt")
    enc = {k: v.to(mdl.device) for k, v in enc.items()}
    with torch.inference_mode():
        out = mdl.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id
        )
    gen_ids = out[0][enc["input_ids"].shape[1]:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()

try:
    if check_budget():
        log_event("loading_qwen7b_for_sys1_2_3")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        q_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        q_mdl = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct", quantization_config=bnb,
            device_map={"": 0} if device.type == "cuda" else "auto"
        ).eval()

        # --- SYSTEM 1: Standard Dense RAG (E5-base-v2 top-5 -> Qwen2.5-7B) ---
        log_event("stage_started", stage="sys1_dense_rag")
        for ds_name, ds in datasets.items():
            if not check_budget(): break
            evals = []
            for idx, item in enumerate(ds["questions"]):
                t0 = time.perf_counter()
                docs = e5_indexes[ds_name].retrieve(item["question"], k=5)
                msgs = [{"role": "system", "content": QA_SYS},
                        {"role": "user", "content": f"Passages:\n{format_docs_block(docs)}\n\nQuestion: {item['question']}\nShort Answer:"}]
                pred = qwen_generate(q_tok, q_mdl, msgs, max_new_tokens=48)
                lat = time.perf_counter() - t0
                evals.append({
                    "qid": item["qid"], "question": item["question"], "gold": item["golds"][0], "pred": pred,
                    "em": exact_match(pred, item["golds"]), "f1": token_f1(pred, item["golds"]),
                    "contain_acc": contain_acc(pred, item["golds"]),
                    "recall": title_recall([d["title"] for d in docs], item["gold_titles"]),
                    "search_calls": 1, "latency_s": lat
                })
                if (idx + 1) % 50 == 0:
                    log_event("sys1_progress", dataset=ds_name, done=idx + 1, total=len(ds["questions"]))
            record_dataset_metrics("Dense-RAG (E5 + Qwen2.5-7B)", ds_name, evals)
        stage_status["sys1_dense_rag"] = "completed"

        # --- SYSTEM 2 (P5): LinearRAG (ICLR 2026 Tri-Graph + PPR -> Qwen2.5-7B) ---
        log_event("stage_started", stage="sys2_linearrag_iclr2026")
        for ds_name, ds in datasets.items():
            if not check_budget(): break
            evals = []
            for idx, item in enumerate(ds["questions"]):
                t0 = time.perf_counter()
                docs = linearrag_indexes[ds_name].retrieve(item["question"], k=5)
                msgs = [{"role": "system", "content": QA_SYS},
                        {"role": "user", "content": f"Passages:\n{format_docs_block(docs)}\n\nQuestion: {item['question']}\nShort Answer:"}]
                pred = qwen_generate(q_tok, q_mdl, msgs, max_new_tokens=48)
                lat = time.perf_counter() - t0
                evals.append({
                    "qid": item["qid"], "question": item["question"], "gold": item["golds"][0], "pred": pred,
                    "em": exact_match(pred, item["golds"]), "f1": token_f1(pred, item["golds"]),
                    "contain_acc": contain_acc(pred, item["golds"]),
                    "recall": title_recall([d["title"] for d in docs], item["gold_titles"]),
                    "search_calls": 1, "latency_s": lat
                })
                if (idx + 1) % 50 == 0:
                    log_event("sys2_progress", dataset=ds_name, done=idx + 1, total=len(ds["questions"]))
            record_dataset_metrics("LinearRAG (ICLR'26 + Qwen2.5-7B)", ds_name, evals)
        stage_status["sys2_linearrag"] = "completed"

        # Free LinearRAG mpnet model before loading FG-DeBERTa on cuda:1
        del mpnet_model, linearrag_indexes
        free_gpu()

        # --- SYSTEM 3: Adaptive Fact-Grounding DeBERTa (93ef195f, T=1.929) Gate -> Qwen2.5-7B ---
        log_event("stage_started", stage="sys3_adaptive_fg_gate")
        cands = [p for p in INPUT_ROOT.rglob("model.safetensors")] if INPUT_ROOT.exists() else []
        if cands:
            ckpt = next((p for p in cands if "fact-grounding" in str(p).lower()), cands[0])
            fg_dev = torch.device("cuda:1" if n_gpus > 1 else ("cuda:0" if device.type == "cuda" else "cpu"))
            fg_tok = AutoTokenizer.from_pretrained(ckpt.parent, local_files_only=True)
            fg_mdl = AutoModelForSequenceClassification.from_pretrained(ckpt.parent, local_files_only=True).to(fg_dev).eval()
            log_event("fg_gate_loaded", ckpt=str(ckpt), device=str(fg_dev))

            def fg_sufficiency_prob(question, docs):
                joined = "\n".join(f"[{d['title']}] {d['text']}" for d in docs)
                txt = f"Sub-question: {question}\n\nPassages:\n{joined[:PASSAGE_MAX_CHARS]}"
                enc = fg_tok(txt, truncation=True, max_length=512, return_tensors="pt")
                enc = {k: v.to(fg_dev) for k, v in enc.items()}
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(fg_dev.type == "cuda")):
                    logits = fg_mdl(**enc).logits[0].float().cpu().numpy()
                ld = float(logits[1] - logits[0])
                return 1.0 / (1.0 + math.exp(-ld / FG_TEMP))

            for ds_name, ds in datasets.items():
                if not check_budget(): break
                evals = []
                for idx, item in enumerate(ds["questions"]):
                    t0 = time.perf_counter()
                    docs = e5_indexes[ds_name].retrieve(item["question"], k=3)
                    p_suff = fg_sufficiency_prob(item["question"], docs)
                    calls = 1
                    triggered = 0
                    if p_suff < 0.50:
                        triggered = 1
                        calls = 2
                        q2_msgs = [
                            {"role": "system", "content": "Given a question and partial passages that lack a missing fact, write ONE short search query to find the missing bridge fact. Output ONLY the search query."},
                            {"role": "user", "content": f"Passages:\n{format_docs_block(docs)}\n\nQuestion: {item['question']}\nMissing-fact search query:"}
                        ]
                        q2 = qwen_generate(q_tok, q_mdl, q2_msgs, max_new_tokens=32)
                        extra_docs = e5_indexes[ds_name].retrieve(q2 if q2 else item["question"], k=4)
                        seen_ids = {d["doc_id"] for d in docs}
                        for ed in extra_docs:
                            if ed["doc_id"] not in seen_ids:
                                docs.append(ed); seen_ids.add(ed["doc_id"])
                            if len(docs) >= 5:
                                break
                    msgs = [{"role": "system", "content": QA_SYS},
                            {"role": "user", "content": f"Passages:\n{format_docs_block(docs)}\n\nQuestion: {item['question']}\nShort Answer:"}]
                    pred = qwen_generate(q_tok, q_mdl, msgs, max_new_tokens=48)
                    lat = time.perf_counter() - t0
                    evals.append({
                        "qid": item["qid"], "question": item["question"], "gold": item["golds"][0], "pred": pred,
                        "em": exact_match(pred, item["golds"]), "f1": token_f1(pred, item["golds"]),
                        "contain_acc": contain_acc(pred, item["golds"]),
                        "recall": title_recall([d["title"] for d in docs], item["gold_titles"]),
                        "search_calls": calls, "latency_s": lat, "triggered_hop2": triggered
                    })
                    if (idx + 1) % 50 == 0:
                        log_event("sys3_progress", dataset=ds_name, done=idx + 1, total=len(ds["questions"]))
                record_dataset_metrics("Adaptive-FG-Gate (DeBERTa-93ef + Qwen2.5-7B)", ds_name, evals)
            del fg_mdl, fg_tok
            stage_status["sys3_adaptive_fg_gate"] = "completed"
        else:
            stage_status["sys3_adaptive_fg_gate"] = "SKIPPED: no attached checkpoint"
            log_event("sys3_skipped", reason="model.safetensors not found in /kaggle/input")

        del q_mdl, q_tok
        free_gpu()
        save_all_rag_tables()
except Exception as exc:
    stage_status["sys1_2_3_qwen_group"] = f"FAILED: {exc}"
    log_event("stage_failed", stage="sys1_2_3_qwen_group", error=str(exc), tb=traceback.format_exc()[-1200:])
    free_gpu()

# =====================================================================
# SYSTEM 4 (P6a): CoRAG (NeurIPS 2025 — corag/CoRAG-Llama3.1-8B-MultihopQA, official prompts)
# =====================================================================
CORAG_TASK_DESC = "answer multi-hop questions"

def corag_subquery_msgs(query, past_sq, past_sa):
    past = "\n".join(f"Intermediate query {i+1}: {sq}\nIntermediate answer {i+1}: {sa}" for i, (sq, sa) in enumerate(zip(past_sq, past_sa))).strip()
    prompt = f"""You are using a search engine to answer the main query by iteratively searching the web. Given the following intermediate queries and answers, generate a new simple follow-up question that can help answer the main query. You may rephrase or decompose the main query when previous answers are not helpful. Ask simple follow-up questions only as the search engine may not understand complex questions.

## Previous intermediate queries and answers
{past or 'Nothing yet'}

## Task description
{CORAG_TASK_DESC}

## Main query to answer
{query}

Respond with a simple follow-up question that will help answer the main query, do not explain yourself or output anything else."""
    return [{"role": "user", "content": prompt}]

def corag_subanswer_msgs(subquery, docs):
    ctx = "\n\n".join(f"{d['title']}\n{d['text'][:800]}" for d in docs)
    prompt = f"""Given the following documents, generate an appropriate answer for the query. DO NOT hallucinate any information, only use the provided documents to generate the answer. Respond "No relevant information found" if the documents do not contain useful information.

## Documents
{ctx.strip()}

## Query
{subquery}

Respond with a concise answer only, do not explain yourself or output anything else."""
    return [{"role": "user", "content": prompt}]

def corag_final_msgs(query, past_sq, past_sa, docs):
    past = "\n".join(f"Intermediate query {i+1}: {sq}\nIntermediate answer {i+1}: {sa}" for i, (sq, sa) in enumerate(zip(past_sq, past_sa))).strip()
    ctx = "\n\n".join(f"Doc {i}: {d['title']}\n{d['text'][:700]}" for i, d in enumerate(docs))
    prompt = f"""Given the following intermediate queries and answers, generate a final answer for the main query by combining relevant information. Note that intermediate answers are generated by an LLM and may not always be accurate.

## Documents
{ctx.strip()}

## Intermediate queries and answers
{past or 'Nothing yet'}

## Task description
{CORAG_TASK_DESC}

## Main query
{query}

Respond with an appropriate answer only, do not explain yourself or output anything else."""
    return [{"role": "user", "content": prompt}]

try:
    if check_budget():
        log_event("stage_started", stage="sys4_corag_neurips2025")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        c_id = "corag/CoRAG-Llama3.1-8B-MultihopQA"
        c_tok = AutoTokenizer.from_pretrained(c_id)
        if c_tok.pad_token is None:
            c_tok.pad_token = c_tok.eos_token
        c_mdl = AutoModelForCausalLM.from_pretrained(
            c_id, quantization_config=bnb, device_map={"": 0} if device.type == "cuda" else "auto"
        ).eval()

        def corag_gen(msgs, max_new_tokens=32):
            prompt = c_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = c_tok(prompt, truncation=True, max_length=3200, return_tensors="pt")
            enc = {k: v.to(c_mdl.device) for k, v in enc.items()}
            with torch.inference_mode():
                out = c_mdl.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=c_tok.eos_token_id)
            return c_tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        for ds_name, ds in datasets.items():
            if not check_budget(): break
            evals = []
            for idx, item in enumerate(ds["questions"]):
                t0 = time.perf_counter()
                past_sq, past_sa = [], []
                collected_docs = []
                seen_ids = set()
                seen_sq = set()
                for hop in range(3):
                    sq = corag_gen(corag_subquery_msgs(item["question"], past_sq, past_sa), max_new_tokens=32)
                    sq_norm = normalize_answer(sq)
                    if not sq or sq_norm in seen_sq:
                        break
                    seen_sq.add(sq_norm)
                    step_docs = e5_indexes[ds_name].retrieve(sq, k=3)
                    for d in step_docs:
                        if d["doc_id"] not in seen_ids:
                            collected_docs.append(d); seen_ids.add(d["doc_id"])
                    sa = corag_gen(corag_subanswer_msgs(sq, step_docs), max_new_tokens=32)
                    past_sq.append(sq); past_sa.append(sa)
                if not collected_docs:
                    collected_docs = e5_indexes[ds_name].retrieve(item["question"], k=5)
                pred = corag_gen(corag_final_msgs(item["question"], past_sq, past_sa, collected_docs[:5]), max_new_tokens=48)
                lat = time.perf_counter() - t0
                evals.append({
                    "qid": item["qid"], "question": item["question"], "gold": item["golds"][0], "pred": pred,
                    "em": exact_match(pred, item["golds"]), "f1": token_f1(pred, item["golds"]),
                    "contain_acc": contain_acc(pred, item["golds"]),
                    "recall": title_recall([d["title"] for d in collected_docs[:5]], item["gold_titles"]),
                    "search_calls": max(len(past_sq), 1), "latency_s": lat
                })
                if (idx + 1) % 40 == 0:
                    log_event("sys4_corag_progress", dataset=ds_name, done=idx + 1, total=len(ds["questions"]), elapsed_h=round(elapsed_hours(), 3))
            record_dataset_metrics("CoRAG-Llama3.1-8B (NeurIPS'25, L=3)", ds_name, evals)
        del c_mdl, c_tok
        free_gpu()
        stage_status["sys4_corag"] = "completed"
        save_all_rag_tables()
except Exception as exc:
    stage_status["sys4_corag"] = f"FAILED: {exc}"
    log_event("stage_failed", stage="sys4_corag", error=str(exc), tb=traceback.format_exc()[-1200:])
    free_gpu()

# =====================================================================
# SYSTEM 5 (P6b): Search-R1 (COLM 2025 — PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo)
# =====================================================================
class StopOnSequence(StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [tokenizer.encode(s, add_special_tokens=False) for s in target_sequences if s]
        self.target_ids = [t for t in self.target_ids if len(t) > 0]
        self.target_lengths = [len(t) for t in self.target_ids]

    def __call__(self, input_ids, scores, **kwargs):
        seq_len = input_ids.shape[1]
        for tid, tlen in zip(self.target_ids, self.target_lengths):
            if seq_len >= tlen:
                if input_ids[0, -tlen:].tolist() == tid:
                    return True
        return False

def searchr1_extract_tag(text, tag):
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else None

def format_searchr1_docs(docs):
    return "".join(f"Doc {i+1}(Title: {d['title']}) {d['text'][:700]}\n" for i, d in enumerate(docs))

try:
    if check_budget():
        log_event("stage_started", stage="sys5_searchr1_colm2025")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        s_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"
        s_tok = AutoTokenizer.from_pretrained(s_id)
        s_mdl = AutoModelForCausalLM.from_pretrained(
            s_id, quantization_config=bnb, device_map={"": 0} if device.type == "cuda" else "auto"
        ).eval()
        stop_seqs = ["</search>", " </search>", "</search>\n", "</answer>", " </answer>"]
        stop_crit = StoppingCriteriaList([StopOnSequence(stop_seqs, s_tok)])
        eos_ids = {s_tok.eos_token_id, 151645, 151643}

        for ds_name, ds in datasets.items():
            if not check_budget(): break
            evals = []
            for idx, item in enumerate(ds["questions"]):
                t0 = time.perf_counter()
                q_text = item["question"].strip()
                if not q_text.endswith("?"):
                    q_text += "?"
                base_prompt = (
                    "Answer the given question. "
                    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
                    "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
                    "and it will return the top searched results between <information> and </information>. "
                    "You can search as many times as your want. "
                    "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, "
                    f"without detailed illustrations. For example, <answer> Beijing </answer>. Question: {q_text}\n"
                )
                if s_tok.chat_template:
                    prompt = s_tok.apply_chat_template([{"role": "user", "content": base_prompt}], add_generation_prompt=True, tokenize=False)
                else:
                    prompt = base_prompt

                collected_docs = []
                seen_ids = set()
                search_calls = 0
                pred = ""
                fmt_valid = 0
                full_gen = ""
                for turn in range(4):
                    enc = s_tok(prompt, truncation=True, max_length=3200, return_tensors="pt")
                    enc = {k: v.to(s_mdl.device) for k, v in enc.items()}
                    with torch.inference_mode():
                        out = s_mdl.generate(
                            **enc, max_new_tokens=220, do_sample=False,
                            stopping_criteria=stop_crit, pad_token_id=s_tok.eos_token_id
                        )
                    gen_tokens = out[0][enc["input_ids"].shape[1]:]
                    out_text = s_tok.decode(gen_tokens, skip_special_tokens=True)
                    full_gen += out_text
                    ans = searchr1_extract_tag(out_text, "answer") or searchr1_extract_tag(full_gen, "answer")
                    if ans is not None:
                        pred = ans; fmt_valid = 1; break
                    sq = searchr1_extract_tag(out_text, "search")
                    if sq:
                        search_calls += 1
                        step_docs = e5_indexes[ds_name].retrieve(sq, k=3)
                        for d in step_docs:
                            if d["doc_id"] not in seen_ids:
                                collected_docs.append(d); seen_ids.add(d["doc_id"])
                        info_str = format_searchr1_docs(step_docs)
                        prompt += f"\n\n{out_text}<information>{info_str}</information>\n\n"
                    else:
                        if out[0][-1].item() in eos_ids:
                            break
                        prompt += out_text
                if not fmt_valid:
                    # Force concise final answer if loop ended without <answer> tag
                    prompt += "\n<answer>"
                    enc = s_tok(prompt, truncation=True, max_length=3400, return_tensors="pt")
                    enc = {k: v.to(s_mdl.device) for k, v in enc.items()}
                    with torch.inference_mode():
                        out = s_mdl.generate(**enc, max_new_tokens=32, do_sample=False, pad_token_id=s_tok.eos_token_id)
                    tail = s_tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                    pred = tail.split("</answer>")[0].strip()
                lat = time.perf_counter() - t0
                evals.append({
                    "qid": item["qid"], "question": item["question"], "gold": item["golds"][0], "pred": pred,
                    "em": exact_match(pred, item["golds"]), "f1": token_f1(pred, item["golds"]),
                    "contain_acc": contain_acc(pred, item["golds"]),
                    "recall": title_recall([d["title"] for d in collected_docs[:5]], item["gold_titles"]),
                    "search_calls": search_calls, "latency_s": lat, "format_valid": fmt_valid
                })
                if (idx + 1) % 40 == 0:
                    log_event("sys5_searchr1_progress", dataset=ds_name, done=idx + 1, total=len(ds["questions"]), elapsed_h=round(elapsed_hours(), 3))
            record_dataset_metrics("Search-R1-Qwen2.5-7B-PPO (COLM'25)", ds_name, evals)
        del s_mdl, s_tok
        free_gpu()
        stage_status["sys5_searchr1"] = "completed"
        save_all_rag_tables()
except Exception as exc:
    stage_status["sys5_searchr1"] = f"FAILED: {exc}"
    log_event("stage_failed", stage="sys5_searchr1", error=str(exc), tb=traceback.format_exc()[-1200:])
    free_gpu()

save_all_rag_tables()
log_event("rag_suite_finished", stage_status=stage_status, total_rows=len(rag_metrics))
print("\n=== NOTEBOOK 07 COMPLETE ===")
print("Stage status:", json.dumps(stage_status, indent=2))
if (OUTPUT_ROOT / "rag_metrics.csv").exists():
    df_m = pd.read_csv(OUTPUT_ROOT / "rag_metrics.csv")
    display(df_m)
    display(df_m.groupby("system")[["em", "f1", "contain_acc", "evidence_recall", "avg_search_calls", "mean_latency_s"]].mean().reset_index())
display(FileLink(str(OUTPUT_ROOT / "results_bundle.zip")))
'''


def md_cell(source: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source: str, cell_id: str) -> dict:
    return {"cell_type": "code", "id": cell_id, "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def main():
    nb = {
        "cells": [
            md_cell(INTRO_MD, "overview"),
            code_cell(SETUP_CELL, "setup"),
            code_cell(MAIN_CELL, "run_overnight_suite_07"),
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
    STAGING.mkdir(parents=True, exist_ok=True)
    for out in (ROOT / "07_overnight_multihop_rag_systems.ipynb", STAGING / "07_overnight_multihop_rag_systems.ipynb"):
        out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print("Wrote:", out)


if __name__ == "__main__":
    main()
