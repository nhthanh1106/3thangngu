from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "kaggle_five_benchmark_eval"

intro = r"""# Fact-Grounding DeBERTa: Five Evaluation Tracks

This notebook evaluates the **existing fine-tuned checkpoint only**. It does not train, fine-tune, or change the decision threshold. It is set up for a Kaggle T4 run with an 8.5-hour inference wall-time budget.

## Evaluation tracks

| Track | Data and decision being measured | What to conclude |
|---|---|---|
| 1. MuSiQue dev | Official released binary answerability labels; the original sub-question + passages format | Closest task match. **Not independent**: this dev split was used for checkpoint selection. |
| 2. HotpotQA support pairs | All eligible matched pairs from the complete 7,405-question distractor dev split; positives contain the full annotated support set and negatives replace one support fact | Controlled transfer to another multi-hop QA dataset; labels are support-fact-derived proxies, with ambiguous/leaking cases excluded. |
| 3. 2Wiki support-removal pairs | All constructable matched pairs from the full dev split; no 1,000-pair cap | Cross-dataset sensitivity to missing evidence. These are support-fact-derived proxy labels, not original binary labels. |
| 4. FaithEval unanswerable pairs | Original context vs. its edited unanswerable context; each source row is its own pair even when `qid` repeats | Tests context sufficiency under an answerability shift; source is a community mirror adding original contexts. |
| 5. Pairwise sufficiency ranking | Pairwise accuracy / score margin across the matched pairs in tracks 2–4 | **RINSE-style diagnostic only**, not the official RINSE benchmark or a fifth independent corpus. |

## Important interpretation limits

The model was trained on decomposed MuSiQue sub-questions. HotpotQA, 2Wiki, and FaithEval examples use full questions, so their results mix domain transfer with an input-format shift. We use the paper's serialization (`Sub-question: ...\n\nPassages:\n...`), 1,500-character passage cap for newly constructed examples, 512-token truncation, and fixed 0.5 threshold. Do not compare these classifier F1/accuracy values directly to QA exact-match/F1 or generative faithfulness scores from papers. The notebook reports binary classification and pairwise metrics on its explicitly described labels.

The checkpoint comes from the private Kaggle dataset `nguynbquangminh/fact-grounding-checkpoint`; the notebook checks and records the model file's SHA-256 before loading it. MuSiQue dev was used to select the saved checkpoint (best dev F1), so use the other tracks for the more informative transfer diagnostics.

## Resume / recovery

Predictions are appended and flushed to disk after every inference batch; progress state is atomically replaced after each batch. The notebook packages the latest predictions, manifest, metrics, and progress into `resume_bundle.zip` every 10 batches and after each track. If a Kaggle session dies, rerunning in the same session resumes from `/kaggle/working`. For a **new** Kaggle session, download the output `resume_bundle.zip`, attach it as a Kaggle Dataset input, and rerun: the notebook discovers and restores it, then resumes only when the model hash and dataset/config signature match. Mismatched old predictions are preserved with a `.stale-*` suffix. The 1.7 GB checkpoint is never copied into the resume archive; keep it attached as the checkpoint dataset input.

Outputs: `metrics.csv`, `pairwise_ranking_metrics.csv`, `error_cases.csv`, per-track JSONL predictions, `run_manifest.json`, `progress_state.json`, and `resume_bundle.zip` under `/kaggle/working/fact_grounding_five_benchmark_eval/`.

## Sources

- Fact-Grounding Gap paper: [EMNLP 2026](https://aclanthology.org/2026.emnlp-main.834/).
- [MuSiQue](https://github.com/StonyBrookNLP/musique) is evaluated here using the released dev CSV included with the reproduction's training-data dataset.
- [HotpotQA](https://aclanthology.org/D18-1259/) provides sentence-level supporting facts for multi-hop questions.
- [2WikiMultiHopQA](https://aclanthology.org/2020.coling-main.580/) provides evidence information and reasoning paths; the notebook downloads its dev parquet from the [Apache-2.0 Hugging Face mirror](https://huggingface.co/datasets/xanhho/2WikiMultihopQA).
- [FaithEval (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/48404cd9ce03946c6b7177691f3267a1-Abstract-Conference.html) includes unanswerable contexts. This notebook uses a [community mirror with original contexts](https://huggingface.co/datasets/alvinming/FaithEval-unanswerable-v1.0-w-original_context) to form paired decisions; treat it as a diagnostic, not an official leaderboard reproduction.
- Track 5 borrows only the *paired sufficiency comparison idea*; it is not a run of an official RINSE release.
"""

setup = r"""import sys, subprocess, time
NOTEBOOK_STARTED_AT = time.monotonic()
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.57.1", "sentencepiece", "protobuf", "scikit-learn", "pyarrow"
])
print("Runtime setup complete. Wall-time budget: 8.5 hours from this cell.")
"""

main = r'''import os, sys, json, time, math, hashlib, random, shutil, zipfile, warnings, subprocess
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import torch
try:
    import kagglehub
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kagglehub"])
    import kagglehub
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, balanced_accuracy_score, roc_auc_score, average_precision_score, brier_score_loss)
from IPython.display import display, FileLink

# ----- Fixed experiment configuration -----
MAX_RUNTIME_HOURS = 8.5
MAX_LENGTH = 512
PASSAGE_MAX_CHARS = 1500
DECISION_THRESHOLD = 0.5
BATCH_SIZE = 12
EXPECTED_CHECKPOINT_SHA256 = "6BBD13D50ABA7EF63B95D3BFA4B63CBA063D79F9F3AA63371A70519D92551DBD"
CHECKPOINT_DATASET_ID = "nguynbquangminh/fact-grounding-checkpoint"
STARTED_AT = globals().get("NOTEBOOK_STARTED_AT", time.monotonic())

INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working/fact_grounding_five_benchmark_eval")
PREDICTION_ROOT = OUTPUT_ROOT / "predictions"
CACHE_ROOT = OUTPUT_ROOT / "download_cache"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
PREDICTION_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUTPUT_ROOT / "run_log.jsonl"
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
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A killed process can leave one incomplete final line; earlier rows are durable.
                continue
    return rows

def find_input(fragment):
    roots = [p for p in INPUT_ROOT.iterdir() if p.is_dir() and fragment.lower() in p.name.lower()]
    if not roots:
        raise FileNotFoundError(f"Attach Kaggle dataset containing {fragment!r} under {INPUT_ROOT}.")
    return roots[0]

def resolve_kaggle_input(fragment, dataset_handle):
    try:
        root = find_input(fragment)
        print(f"Mounted input resolved: {dataset_handle} -> {root}")
        log_event("kaggle_input_resolved", dataset=dataset_handle, path=str(root), method="mounted")
        return root
    except FileNotFoundError:
        # Some Kaggle API-created kernel versions retain source metadata but do not
        # expose the mount to the execution container. KaggleHub attaches/resolves it.
        print(f"Input mount missing for {dataset_handle}; resolving with KaggleHub.")
        resolved = Path(kagglehub.dataset_download(dataset_handle))
        print(f"KaggleHub dataset path: {dataset_handle} -> {resolved}")
        if not resolved.exists():
            raise FileNotFoundError(f"KaggleHub returned a missing path for {dataset_handle}: {resolved}")
        log_event("kaggle_input_resolved", dataset=dataset_handle, path=str(resolved), method="kagglehub")
        return resolved

def restore_resume_bundle():
    restored = []
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

restore_resume_bundle()
log_event("run_started", time_budget_hours=MAX_RUNTIME_HOURS,
          output_dir=str(OUTPUT_ROOT))
STATE_PATH = OUTPUT_ROOT / "progress_state.json"
state = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {"benchmarks": {}}
state.setdefault("benchmarks", {})

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

def elapsed_hours():
    return (time.monotonic() - STARTED_AT) / 3600.0

def check_time_budget():
    return elapsed_hours() < MAX_RUNTIME_HOURS

# ----- Locate and verify the exact saved checkpoint -----
checkpoint_dataset = resolve_kaggle_input("fact-grounding-checkpoint", CHECKPOINT_DATASET_ID)
model_files = list(checkpoint_dataset.rglob("model.safetensors"))
if not model_files:
    raise FileNotFoundError(f"No model.safetensors found inside {checkpoint_dataset}")
checkpoint_file = model_files[0]
checkpoint_dir = checkpoint_file.parent
checkpoint_sha256 = sha256_file(checkpoint_file)
print("Checkpoint dataset:", CHECKPOINT_DATASET_ID)
print("Checkpoint file:", checkpoint_file)
print("Checkpoint bytes:", checkpoint_file.stat().st_size)
print("Checkpoint SHA-256:", checkpoint_sha256)
log_event("checkpoint_verified", dataset=CHECKPOINT_DATASET_ID,
          file=str(checkpoint_file), bytes=checkpoint_file.stat().st_size, sha256=checkpoint_sha256)
if checkpoint_sha256.upper() != EXPECTED_CHECKPOINT_SHA256:
    raise ValueError("Checkpoint hash differs from the expected paper checkpoint. Stop instead of evaluating an unintended model.")
provenance_files = list(checkpoint_dataset.rglob("checkpoint-provenance.json"))
if not provenance_files:
    raise FileNotFoundError("checkpoint-provenance.json missing; cannot verify which saved checkpoint is being evaluated.")
checkpoint_provenance = json.loads(provenance_files[0].read_text(encoding="utf-8"))
best_ckpt = checkpoint_provenance.get("best_checkpoint", {})
if checkpoint_provenance.get("model") != "microsoft/deberta-v3-large" or best_ckpt.get("global_step") != 11656:
    raise ValueError(f"Unexpected checkpoint provenance: {checkpoint_provenance}")
log_event("checkpoint_provenance_verified", source_kernel=checkpoint_provenance.get("source_kernel"),
          global_step=best_ckpt.get("global_step"), epoch=best_ckpt.get("epoch"), dev_f1=best_ckpt.get("dev_f1"),
          run_status=checkpoint_provenance.get("run_status"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir, local_files_only=True)
model.to(device).eval()
print("Loaded existing classifier. No training or threshold tuning will run.")
log_event("model_loaded", device=str(device), batch_size=BATCH_SIZE,
          max_length=MAX_LENGTH, threshold=DECISION_THRESHOLD)

def serialize_input(question, passage):
    if isinstance(passage, (list, tuple)):
        passage = "\n".join(str(p) for p in passage)
    passage = str(passage or "")[:PASSAGE_MAX_CHARS]
    return f"Sub-question: {str(question or '').strip()}\n\nPassages:\n{passage}"

def canonical_text(row):
    return str(row.get("input_text") or serialize_input(row.get("question", ""), row.get("passages", "")))

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
            with requests.get(url, stream=True, timeout=(30, 180), headers={"User-Agent": "fact-grounding-eval/1.0"}) as response:
                response.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
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

# ----- Prepare benchmark examples -----
benchmarks = {}
benchmark_sources = {}

# 1) MuSiQue released dev labels in the exact training/evaluation serialization.
training_data_root = resolve_kaggle_input("fact-grounding-training-data", "nguynbquangminh/fact-grounding-training-data")
dev_csv = next(training_data_root.rglob("deberta_fg_dev.csv"), None)
if dev_csv is None:
    raise FileNotFoundError("deberta_fg_dev.csv missing from fact-grounding-training-data")
dev_df = pd.read_csv(dev_csv)
musique_rows = []
for idx, r in dev_df.iterrows():
    raw_text = str(r["text"])
    musique_rows.append({
        "example_id": f"musique-dev-{idx}", "pair_id": None,
        "question": "", "passages": "", "input_text": raw_text,
        "label": int(r["label"]), "label_source": "Official released MuSiQue dev binary label",
        "construction_method": "Original dev example; exact saved text serialization."
    })
benchmarks["01_musique_dev"] = musique_rows
benchmark_sources["01_musique_dev"] = {"file": str(dev_csv), "sha256": sha256_file(dev_csv), "caveat": "Dev was used for checkpoint selection; not independent."}

# 2) HotpotQA support pairs from all eligible questions in the complete official dev distractor split.
eval_data_root = resolve_kaggle_input("fact-grounding-eval-data", "nguynbquangminh/fact-grounding-eval-data")
prepared_jsonl = next(eval_data_root.rglob("prepared_eval_examples.jsonl"), None)
if prepared_jsonl is None:
    raise FileNotFoundError("prepared_eval_examples.jsonl missing from fact-grounding-eval-data")
hotpot_rows = []
with open(prepared_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        if ex.get("dataset") != "HotpotQA":
            continue
        exid = str(ex.get("id", len(hotpot_rows)))
        suffix = exid.rsplit("-", 1)[-1]
        pair_id = exid.rsplit("-", 1)[0] if suffix in {"positive", "negative"} else None
        passages = ex.get("passages", [])
        hotpot_rows.append({
            "example_id": exid, "pair_id": pair_id,
            "question": str(ex.get("question", "")), "passages": passages,
            "input_text": serialize_input(ex.get("question", ""), passages),
            "label": int(ex["label"]), "label_source": str(ex.get("label_source", "Prepared supporting-fact pair")),
            "construction_method": str(ex.get("construction_method", "Prepared HotpotQA contrast pair."))
        })
benchmarks["02_hotpotqa_pairs"] = hotpot_rows
benchmark_sources["02_hotpotqa_pairs"] = {"file": str(prepared_jsonl), "sha256": sha256_file(prepared_jsonl), "n_examples": len(hotpot_rows), "n_source_questions": len(hotpot_rows) // 2, "caveat": "All eligible rows from the complete HotpotQA distractor dev split; support-fact-derived proxy labels; full question format differs from training sub-questions."}

# 3) 2Wiki: deterministic positive/negative evidence pairs from the official dev parquet mirror.
twowiki_path = download_public_file(
    "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/dev.parquet?download=true",
    CACHE_ROOT / "2wiki_dev.parquet"
)
twowiki_df = pd.read_parquet(twowiki_path)

def context_documents(value):
    value = read_jsonish(value)
    docs = []
    if isinstance(value, dict):
        titles = value.get("title", value.get("titles", []))
        sentences = value.get("sentences", value.get("sentence", []))
        for title, sents in zip(titles or [], sentences or []):
            docs.append((str(title), [str(s) for s in (sents or [])]))
    elif isinstance(value, (list, tuple)):
        for item in value:
            item = read_jsonish(item)
            if isinstance(item, dict):
                title = item.get("title", "")
                sents = item.get("sentences", item.get("sentence", []))
                docs.append((str(title), [str(s) for s in (sents or [])]))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                sents = read_jsonish(item[1])
                if isinstance(sents, str):
                    sents = [sents]
                docs.append((str(item[0]), [str(s) for s in (sents or [])]))
    return docs

def support_tuples(value):
    value = read_jsonish(value)
    facts = []
    if isinstance(value, dict):
        titles = value.get("title", value.get("titles", []))
        indices = value.get("sent_id", value.get("sentence_id", value.get("sent_ids", [])))
        for title, idx in zip(titles or [], indices or []):
            try: facts.append((str(title), int(idx)))
            except Exception: pass
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try: facts.append((str(item[0]), int(item[1])))
                except Exception: pass
            elif isinstance(item, dict):
                try: facts.append((str(item.get("title", "")), int(item.get("sent_id", item.get("sentence_id")))))
                except Exception: pass
    return facts

def render_docs(docs):
    return "\n".join(f"[{title}] {sentence}" for title, sents in docs for sentence in sents)

twowiki_records = twowiki_df.to_dict("records")
twowiki_order = list(range(len(twowiki_records)))
random.Random(42).shuffle(twowiki_order)
twowiki_rows = []
for idx in twowiki_order:
    ex = twowiki_records[idx]
    question = str(ex.get("question", ""))
    docs = context_documents(ex.get("context"))
    supports = support_tuples(ex.get("supporting_facts"))
    if not question or not docs or not supports:
        continue
    positive_passage = render_docs(docs)
    answer = str(ex.get("answer", "")).strip()
    chosen_negative = None
    for title, sent_idx in supports:
        neg_docs = []
        removed = False
        for doc_title, sentences in docs:
            new_sents = list(sentences)
            if not removed and doc_title == title and 0 <= sent_idx < len(new_sents):
                new_sents.pop(sent_idx)
                removed = True
            neg_docs.append((doc_title, new_sents))
        if not removed:
            continue
        negative_passage = render_docs(neg_docs)
        # Keep only cases where the answer string is not explicitly left in the negative evidence.
        if answer and answer.casefold() in negative_passage.casefold():
            continue
        if serialize_input(question, positive_passage) == serialize_input(question, negative_passage):
            continue
        chosen_negative = negative_passage
        break
    if chosen_negative is None:
        continue
    pair_id = str(ex.get("_id", ex.get("id", idx)))
    for suffix, passage, label in (("positive", positive_passage, 1), ("negative", chosen_negative, 0)):
        twowiki_rows.append({
            "example_id": f"2wiki-{pair_id}-{suffix}", "pair_id": f"2wiki-{pair_id}",
            "question": question, "passages": passage,
            "input_text": serialize_input(question, passage), "label": label,
            "label_source": "Support-fact-derived proxy label",
            "construction_method": "Positive uses supplied context; negative deletes one supporting sentence and is retained only when the literal answer is absent and the model inputs differ."
        })
benchmarks["03_2wiki_support_pairs"] = twowiki_rows
benchmark_sources["03_2wiki_support_pairs"] = {"file": str(twowiki_path), "sha256": sha256_file(twowiki_path), "n_examples": len(twowiki_rows), "n_source_questions": len(twowiki_rows) // 2, "source": "xanhho/2WikiMultihopQA dev.parquet (official mirror; Apache-2.0 metadata)", "caveat": "All constructable dev examples; synthetic support-deletion contrasts; proxy labels, not an official binary benchmark."}

# 4) FaithEval unanswerable: original context is answerable; edited context is unanswerable.
faith_path = download_public_file(
    "https://huggingface.co/datasets/alvinming/FaithEval-unanswerable-v1.0-w-original_context/resolve/main/data/train-00000-of-00001.parquet?download=true",
    CACHE_ROOT / "faitheval_unanswerable.parquet"
)
faith_df = pd.read_parquet(faith_path)
faith_rows = []
for idx, ex in enumerate(faith_df.to_dict("records")):
    qid = str(ex.get("qid", idx))
    question = str(ex.get("question", ""))
    original = ex.get("original_context", "")
    edited = ex.get("context", "")
    if not question or not str(original).strip() or not str(edited).strip():
        continue
    # qid can repeat when the same question/context has multiple distinct edits.
    # Preserve each edited source row as a separate pair instead of merging IDs.
    pair_id = f"faitheval-{qid}-row-{idx}"
    for suffix, passage, label in (("answerable", original, 1), ("unanswerable", edited, 0)):
        faith_rows.append({
            "example_id": f"{pair_id}-{suffix}", "pair_id": pair_id,
            "question": question, "passages": str(passage),
            "input_text": serialize_input(question, passage), "label": label,
            "label_source": "FaithEval original-context vs. edited-unanswerable contrast",
            "construction_method": "Paired original_context (label 1) and modified context (label 0) from one community-mirror source row; repeated qid rows with different edits remain distinct; not the official FaithEval leaderboard protocol."
        })
benchmarks["04_faitheval_pairs"] = faith_rows
benchmark_sources["04_faitheval_pairs"] = {"file": str(faith_path), "sha256": sha256_file(faith_path), "source": "Community mirror adding original_context to FaithEval unanswerable-v1.0", "caveat": "Original/edited paired diagnostic; full-question format and community mirror."}

for name, rows in benchmarks.items():
    if not rows:
        raise ValueError(f"Prepared benchmark {name} has no examples; inspect the source schema.")
    print(f"{name}: {len(rows):,} examples | labels={pd.Series([r['label'] for r in rows]).value_counts().to_dict()}")
    log_event("benchmark_prepared", benchmark=name, n_examples=len(rows), source=benchmark_sources[name])

# Signatures bind resume data to the exact checkpoint, input examples, and evaluation configuration.
benchmark_signatures = {}
for name, rows in benchmarks.items():
    h = hashlib.sha256()
    h.update(checkpoint_sha256.encode())
    h.update(json.dumps({"max_length": MAX_LENGTH, "passage_max_chars": PASSAGE_MAX_CHARS,
                         "threshold": DECISION_THRESHOLD, "batch_size": BATCH_SIZE,
                         "label_mapping": {0: "NOT-ANSWERABLE", 1: "ANSWERABLE"}}, sort_keys=True).encode())
    h.update(rows_signature(rows).encode())
    benchmark_signatures[name] = h.hexdigest()

manifest = {
    "paper": "Diagnosing the Fact-Grounding Gap in Multi-Hop Question Answering (EMNLP 2026)",
    "paper_url": "https://aclanthology.org/2026.emnlp-main.834/",
    "checkpoint_dataset": CHECKPOINT_DATASET_ID,
    "checkpoint_file": str(checkpoint_file),
    "checkpoint_sha256": checkpoint_sha256,
    "checkpoint_bytes": checkpoint_file.stat().st_size,
    "checkpoint_provenance_file": str(provenance_files[0]),
    "checkpoint_provenance": checkpoint_provenance,
    "base_protocol": {"max_length": MAX_LENGTH, "new_input_passage_max_chars": PASSAGE_MAX_CHARS,
                      "threshold": DECISION_THRESHOLD, "label_mapping": {"0": "NOT-ANSWERABLE", "1": "ANSWERABLE"}},
    "time_budget_hours": MAX_RUNTIME_HOURS,
    "benchmark_sources": benchmark_sources,
    "benchmark_signatures": benchmark_signatures,
    "started_at_utc": utc_now()
}
atomic_json(OUTPUT_ROOT / "run_manifest.json", manifest)

# ----- Metrics and durable, resumable inference -----
def expected_calibration_error(y, p, bins=15):
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mask.any():
            total += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)

def calculate_metrics(records, expected_count):
    y = np.asarray([int(r["label"]) for r in records], dtype=int)
    p = np.asarray([float(r["p_answerable"]) for r in records], dtype=float)
    pred = (p >= DECISION_THRESHOLD).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    out = {
        "n_expected": int(expected_count), "n_scored": int(len(records)),
        "complete": bool(len(records) == expected_count),
        "accuracy": float(accuracy_score(y, pred)) if len(y) else np.nan,
        "precision_answerable": float(precision_score(y, pred, zero_division=0)) if len(y) else np.nan,
        "recall_answerable": float(recall_score(y, pred, zero_division=0)) if len(y) else np.nan,
        "f1_answerable": float(f1_score(y, pred, zero_division=0)) if len(y) else np.nan,
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)) if len(y) else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(set(y)) > 1 else np.nan,
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        "mean_inference_ms": float(np.mean([r["inference_ms"] for r in records])) if records else np.nan,
        "p95_inference_ms": float(np.percentile([r["inference_ms"] for r in records], 95)) if records else np.nan,
        "brier": float(brier_score_loss(y, p)) if len(y) else np.nan,
        "ece_15_bins": expected_calibration_error(y, p) if len(y) else np.nan
    }
    out["auroc"] = float(roc_auc_score(y, p)) if len(set(y)) > 1 else np.nan
    out["pr_auc_average_precision"] = float(average_precision_score(y, p)) if len(set(y)) > 1 else np.nan
    return out

def write_partial_outputs(metric_rows):
    pd.DataFrame(metric_rows).to_csv(OUTPUT_ROOT / "metrics.csv", index=False)

metric_rows = []
timed_out = False
completed_names = []
for name, rows in benchmarks.items():
    signature = benchmark_signatures[name]
    pred_path = PREDICTION_ROOT / f"{name}.jsonl"
    prior = state["benchmarks"].get(name, {})
    if pred_path.exists() and prior.get("signature") != signature:
        stale = pred_path.with_name(pred_path.stem + ".stale-" + datetime.now().strftime("%Y%m%dT%H%M%S") + pred_path.suffix)
        os.replace(pred_path, stale)
        print(f"Signature changed; preserved old predictions at {stale.name}")
    existing = safe_read_jsonl(pred_path) if state["benchmarks"].get(name, {}).get("signature") == signature else []
    done_ids = {str(r["example_id"]) for r in existing}
    rows_by_id = {str(r["example_id"]): r for r in rows}
    # Drop any unexpected rows, though signatures normally make this unnecessary.
    done_ids &= set(rows_by_id)
    remaining = [r for r in rows if str(r["example_id"]) not in done_ids]
    print(f"\n{name}: {len(done_ids):,}/{len(rows):,} already durable; {len(remaining):,} remaining.")
    log_event("benchmark_started", benchmark=name, already_durable=len(done_ids),
              remaining=len(remaining), expected=len(rows), signature=signature)
    n_batches = 0
    for start in range(0, len(remaining), BATCH_SIZE):
        if not check_time_budget():
            timed_out = True
            print(f"Reached {MAX_RUNTIME_HOURS}h budget before the next batch; saving partial outputs.")
            log_event("runtime_budget_reached", benchmark=name, rows_done=len(done_ids), rows_expected=len(rows))
            break
        batch = remaining[start:start + BATCH_SIZE]
        texts = [canonical_text(r) for r in batch]
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
            probs = torch.softmax(logits.float(), dim=-1)[:, 1].detach().cpu().numpy()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / max(len(batch), 1)
        with open(pred_path, "a", encoding="utf-8") as f:
            for row, prob in zip(batch, probs):
                rec = {
                    "dataset": name, "example_id": str(row["example_id"]), "pair_id": row.get("pair_id"),
                    "question": str(row.get("question", "")), "label": int(row["label"]),
                    "label_source": row.get("label_source", ""), "construction_method": row.get("construction_method", ""),
                    "p_answerable": float(prob), "prediction": int(float(prob) >= DECISION_THRESHOLD),
                    "threshold": DECISION_THRESHOLD, "inference_ms": float(elapsed_ms),
                    "input_preview": canonical_text(row)[:2200]
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush(); os.fsync(f.fileno())
        done_ids.update(str(r["example_id"]) for r in batch)
        n_batches += 1
        state["benchmarks"][name] = {"signature": signature, "rows_done": len(done_ids),
            "rows_expected": len(rows), "last_batch_at_utc": utc_now(), "prediction_file": str(pred_path)}
        state["last_updated_utc"] = utc_now()
        atomic_json(STATE_PATH, state)
        log_event("batch_durable", benchmark=name, rows_done=len(done_ids), rows_expected=len(rows),
                  batch_size=len(batch), mean_inference_ms_per_example=round(float(elapsed_ms), 3),
                  elapsed_hours=round(elapsed_hours(), 4))
        if n_batches % 10 == 0:
            write_resume_bundle()
            print(f"  durable checkpoint: {len(done_ids):,}/{len(rows):,} rows | elapsed {elapsed_hours():.2f}h")
    current_records = safe_read_jsonl(pred_path) if state["benchmarks"].get(name, {}).get("signature") == signature else []
    current_records = [r for r in current_records if str(r.get("example_id")) in rows_by_id]
    if current_records:
        metric = {"benchmark": name, **calculate_metrics(current_records, len(rows))}
        metric_rows.append(metric)
        write_partial_outputs(metric_rows)
    complete = len(done_ids) == len(rows)
    state["benchmarks"][name] = {"signature": signature, "rows_done": len(done_ids),
        "rows_expected": len(rows), "complete": complete, "last_updated_utc": utc_now(),
        "prediction_file": str(pred_path)}
    state["last_updated_utc"] = utc_now()
    atomic_json(STATE_PATH, state)
    log_event("benchmark_finished", benchmark=name, complete=complete,
              rows_done=len(done_ids), rows_expected=len(rows))
    write_resume_bundle()
    if complete:
        completed_names.append(name)
        print(f"Completed {name}: {len(done_ids):,} examples.")
    else:
        print(f"Partial {name}: {len(done_ids):,}/{len(rows):,} examples.")
    if timed_out:
        break

# ----- Paired ranking diagnostic (derived from tracks 2–4; not a separate corpus) -----
prediction_records = []
for pth in sorted(PREDICTION_ROOT.glob("*.jsonl")):
    if ".stale-" not in pth.name:
        prediction_records.extend(safe_read_jsonl(pth))
pred_df = pd.DataFrame(prediction_records)
pair_metrics = []
if not pred_df.empty and "pair_id" in pred_df:
    paired = pred_df[pred_df["pair_id"].notna()]
    for dataset, group in paired.groupby("dataset"):
        pair_scores = []
        for pair_id, pair in group.groupby("pair_id"):
            pos = pair[pair["label"] == 1]
            neg = pair[pair["label"] == 0]
            if len(pos) == 1 and len(neg) == 1:
                delta = float(pos.iloc[0]["p_answerable"] - neg.iloc[0]["p_answerable"])
                pair_scores.append({"pair_id": pair_id, "delta": delta, "correct": 1.0 if delta > 0 else (0.5 if delta == 0 else 0.0)})
        if pair_scores:
            pair_metrics.append({"benchmark": dataset, "n_pairs": len(pair_scores),
                "pairwise_accuracy_ties_half": float(np.mean([x["correct"] for x in pair_scores])),
                "mean_positive_minus_negative_score": float(np.mean([x["delta"] for x in pair_scores]))})
    all_pairs = [r for r in pair_metrics]
    if all_pairs:
        all_scores = []
        for dataset, group in paired.groupby("dataset"):
            for pair_id, pair in group.groupby("pair_id"):
                pos = pair[pair["label"] == 1]; neg = pair[pair["label"] == 0]
                if len(pos) == 1 and len(neg) == 1:
                    delta = float(pos.iloc[0]["p_answerable"] - neg.iloc[0]["p_answerable"])
                    all_scores.append((1.0 if delta > 0 else (0.5 if delta == 0 else 0.0), delta))
        pair_metrics.append({"benchmark": "ALL_MATCHED_PAIRS", "n_pairs": len(all_scores),
            "pairwise_accuracy_ties_half": float(np.mean([x[0] for x in all_scores])),
            "mean_positive_minus_negative_score": float(np.mean([x[1] for x in all_scores]))})
pairwise_df = pd.DataFrame(pair_metrics)
pairwise_df.to_csv(OUTPUT_ROOT / "pairwise_ranking_metrics.csv", index=False)

if not pred_df.empty:
    errors = pred_df[pred_df["label"].astype(int) != pred_df["prediction"].astype(int)].copy()
    errors.to_csv(OUTPUT_ROOT / "error_cases.csv", index=False)
    # Recompute metrics from all durable records, marking partial tracks as incomplete.
    metric_rows = []
    for name, rows in benchmarks.items():
        path = PREDICTION_ROOT / f"{name}.jsonl"
        recs = safe_read_jsonl(path)
        recs = [r for r in recs if str(r.get("example_id")) in {str(x["example_id"]) for x in rows}]
        if recs:
            metric_rows.append({"benchmark": name, **calculate_metrics(recs, len(rows))})
    write_partial_outputs(metric_rows)

manifest["finished_at_utc"] = utc_now()
manifest["elapsed_hours"] = round(elapsed_hours(), 4)
manifest["timed_out_at_budget"] = bool(timed_out)
manifest["completed_benchmarks"] = completed_names
manifest["progress_state"] = state.get("benchmarks", {})
atomic_json(OUTPUT_ROOT / "run_manifest.json", manifest)
atomic_json(STATE_PATH, state)
write_resume_bundle()

print(f"\nElapsed: {elapsed_hours():.2f}h | completed tracks: {completed_names}")
print("Output directory:", OUTPUT_ROOT)
for filename in ["metrics.csv", "pairwise_ranking_metrics.csv", "error_cases.csv", "run_manifest.json", "progress_state.json", "resume_bundle.zip"]:
    path = OUTPUT_ROOT / filename
    if path.exists():
        print(f"- {filename}: {path.stat().st_size:,} bytes")
if (OUTPUT_ROOT / "metrics.csv").exists():
    display(pd.read_csv(OUTPUT_ROOT / "metrics.csv"))
if (OUTPUT_ROOT / "pairwise_ranking_metrics.csv").exists():
    display(pd.read_csv(OUTPUT_ROOT / "pairwise_ranking_metrics.csv"))
display(FileLink(str(OUTPUT_ROOT / "resume_bundle.zip")))
'''

def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "id": "overview", "metadata": {}, "source": source.splitlines(keepends=True)}

def code_cell(source: str, cell_id: str) -> dict:
    return {"cell_type": "code", "id": cell_id, "execution_count": None,
            "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}

notebook = {
    "cells": [md_cell(intro), code_cell(setup, "setup"), code_cell(main, "run_eval")],
    "metadata": {
        "kaggle": {"accelerator": "GPU", "dataSources": [], "isInternetEnabled": True,
                   "language": "python", "sourceType": "notebook"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

ROOT.mkdir(parents=True, exist_ok=True)
STAGING.mkdir(parents=True, exist_ok=True)
notebook_path = ROOT / "04_five_benchmark_eval.ipynb"
staged_notebook_path = STAGING / "04_five_benchmark_eval.ipynb"
for path in (notebook_path, staged_notebook_path):
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")

metadata = {
    "id": "nguynbquangminh/fact-grounding-five-benchmark-evaluation",
    "title": "Fact Grounding Five Benchmark Evaluation",
    "code_file": "04_five_benchmark_eval.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [
        "nguynbquangminh/fact-grounding-checkpoint",
        "nguynbquangminh/fact-grounding-training-data",
        "nguynbquangminh/fact-grounding-eval-data"
    ],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": []
}
(STAGING / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(notebook_path)
print(staged_notebook_path)
print(STAGING / "kernel-metadata.json")
