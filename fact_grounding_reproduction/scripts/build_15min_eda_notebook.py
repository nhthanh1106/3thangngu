import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "kaggle_15min_eda"


def md(source):
    return {
        "cell_type": "markdown",
        "id": hashlib.sha1(source.encode()).hexdigest()[:12],
        "metadata": {},
        "source": source.splitlines(True),
    }


def code(source):
    return {
        "cell_type": "code",
        "id": hashlib.sha1(source.encode()).hexdigest()[:12],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


cells = [
    md("""# Fact-grounding 15-minute classifier pilot

This inference-only run benchmarks the persisted DeBERTa checkpoint on the prepared labeled examples. It evaluates the full input and a question-only diagnostic, then saves batch-level predictions and EDA files under `/kaggle/working/fact_grounding_15min_eda/<run_id>/`.

**Scope:** 300 MuSiQue + 300 HotpotQA + 42 FiQA-2018 examples. No model training, retrieval, or answer generator runs here. It measures classifier quality; it does not reproduce the paper's end-to-end QA accuracy table. SciFact is skipped because its released labels are not answerability labels. The notebook has a 15-minute soft budget and persists each completed inference batch.

Attach the training notebook output `nguynbquangminh/fact-grounding-training-reproduction` as a kernel source and `nguynbquangminh/fact-grounding-eval-data` as a dataset source. This notebook is configured to do that.
"""),
    code(r'''import os, sys, json, time, shutil, subprocess, hashlib, traceback
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError

RUN_STARTED_MONO = time.monotonic()
RUN_STARTED_UTC = datetime.now(timezone.utc).isoformat()
MAX_RUNTIME_SECONDS = 15 * 60
SOFT_STOP_SECONDS = 14 * 60
SAFETY_MARGIN_SECONDS = 60
PINNED_TRANSFORMERS = "4.57.1"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_DIR = Path("/kaggle/working/fact_grounding_15min_eda") / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = RUN_DIR / "run_state.json"
PROGRESS_PATH = RUN_DIR / "progress.jsonl"
STATE = {"run_id": RUN_ID, "started_at_utc": RUN_STARTED_UTC, "events": []}

def elapsed_seconds():
    return time.monotonic() - RUN_STARTED_MONO

def remaining_seconds():
    return max(0.0, MAX_RUNTIME_SECONDS - elapsed_seconds())

def checkpoint(stage, status="complete", **details):
    event = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed_seconds(), 3),
        "stage": stage,
        "status": status,
        **details,
    }
    STATE["events"].append(event)
    STATE["latest"] = event
    STATE["elapsed_seconds"] = round(elapsed_seconds(), 3)
    with PROGRESS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        f.flush()
    STATE_PATH.write_text(json.dumps(STATE, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(event, ensure_ascii=False, default=str), flush=True)

checkpoint("bootstrap", "begin", run_dir=str(RUN_DIR), budget_seconds=MAX_RUNTIME_SECONDS)
try:
    installed = version("transformers")
except PackageNotFoundError:
    installed = None
if installed != PINNED_TRANSFORMERS:
    checkpoint("dependency_install", "begin", existing_transformers=installed, target=PINNED_TRANSFORMERS)
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", f"transformers=={PINNED_TRANSFORMERS}", "sentencepiece", "protobuf", "scikit-learn"],
            check=True,
            timeout=240,
        )
    except Exception as exc:
        checkpoint("dependency_install", "error", error=repr(exc))
        raise
checkpoint("dependency_install", "complete", transformers=PINNED_TRANSFORMERS)
print("RUN_DIR=", RUN_DIR, flush=True)
'''),
    code(r'''import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

if not torch.cuda.is_available():
    checkpoint("environment", "error", reason="CUDA GPU is required for this 15-minute run")
    raise RuntimeError("Enable a Kaggle GPU (Tesla T4).")

DEVICE = torch.device("cuda")
BATCH_SIZE = 16
MAX_LENGTH = 512
THRESHOLD = 0.5
ENVIRONMENT = {
    "python": sys.version,
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "cuda": torch.version.cuda,
    "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "gpu_count": torch.cuda.device_count(),
    "batch_size": BATCH_SIZE,
    "max_length": MAX_LENGTH,
    "threshold": THRESHOLD,
}
checkpoint("environment", "complete", **ENVIRONMENT)
'''),
    code(r'''INPUT_ROOT = Path("/kaggle/input")

def find_eval_root():
    for root in INPUT_ROOT.iterdir():
        if "fact-grounding-eval-data" not in root.name.lower():
            continue
        manifests = list(root.rglob("dataset_manifest.json"))
        examples = list(root.rglob("prepared_eval_examples.jsonl"))
        if manifests and examples:
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            if "sample_counts" in manifest.get("evaluation_data", {}):
                return root, manifests[0], examples[0], manifest
    raise FileNotFoundError("Evaluation data input missing. Attach fact-grounding-eval-data.")

def find_checkpoint():
    candidates = []
    for root in INPUT_ROOT.iterdir():
        for config in root.rglob("config.json"):
            folder = config.parent
            if (folder / "model.safetensors").exists() or (folder / "pytorch_model.bin").exists():
                candidates.append(folder)
    if not candidates:
        raise FileNotFoundError("No model checkpoint found in Kaggle inputs.")
    preferred = [p for p in candidates if p.name == "deberta_fact_grounding_checkpoint"]
    return sorted(preferred or candidates, key=lambda p: len(p.parts))[0]

try:
    EVAL_ROOT, EVAL_MANIFEST_PATH, EXAMPLES_PATH, EVAL_MANIFEST = find_eval_root()
    CHECKPOINT_DIR = find_checkpoint()
    EXAMPLES = [json.loads(line) for line in EXAMPLES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    REQUIRED_FIELDS = {"id", "dataset", "question", "passages", "label"}
    if not EXAMPLES or any(not REQUIRED_FIELDS.issubset(row) for row in EXAMPLES):
        raise ValueError("Prepared examples are empty or missing required fields.")
    if any(int(row["label"]) not in (0, 1) for row in EXAMPLES):
        raise ValueError("Labels must be binary 0/1.")
    DF = pd.DataFrame(EXAMPLES)
    if DF["id"].duplicated().any():
        raise ValueError("Evaluation example IDs are not unique.")
    (RUN_DIR / "evaluation_dataset_manifest.json").write_text(
        EVAL_MANIFEST_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    input_files = []
    for path in sorted(CHECKPOINT_DIR.iterdir()):
        if path.is_file():
            input_files.append({"file": path.name, "bytes": path.stat().st_size})
    pd.DataFrame(input_files).to_csv(RUN_DIR / "checkpoint_inventory.csv", index=False)
    counts = DF.groupby(["dataset", "label"], dropna=False).size().rename("n").reset_index()
    counts.to_csv(RUN_DIR / "input_counts.csv", index=False)
    checkpoint(
        "inputs", "complete", checkpoint_dir=str(CHECKPOINT_DIR),
        eval_root=str(EVAL_ROOT), examples_file=str(EXAMPLES_PATH),
        n_examples=len(DF), counts=counts.to_dict("records"),
        sample_counts_declared=EVAL_MANIFEST.get("evaluation_data", {}).get("sample_counts", {}),
    )
except Exception as exc:
    checkpoint("inputs", "error", error=repr(exc), traceback=traceback.format_exc()[-4000:])
    raise
'''),
    code(r'''if remaining_seconds() < 90:
    checkpoint("model_load", "skipped", reason="less than 90 seconds remain")
    raise TimeoutError("Budget too short to load the model safely.")

model_load_start = time.monotonic()
try:
    TOKENIZER = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR), local_files_only=True)
    MODEL = AutoModelForSequenceClassification.from_pretrained(
        str(CHECKPOINT_DIR), local_files_only=True
    ).to(DEVICE).eval()
    if MODEL.config.label2id.get("ANSWERABLE") != 1 or MODEL.config.label2id.get("NOT-ANSWERABLE") != 0:
        raise ValueError(f"Unexpected label mapping: {MODEL.config.label2id}")
    MODEL_INFO = {
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "architecture": MODEL.config.model_type,
        "parameter_count": sum(p.numel() for p in MODEL.parameters()),
        "label2id": MODEL.config.label2id,
        "num_labels": MODEL.config.num_labels,
        "checkpoint_file_inventory": input_files,
        "load_seconds": round(time.monotonic() - model_load_start, 3),
    }
    (RUN_DIR / "model_metadata.json").write_text(json.dumps(MODEL_INFO, indent=2, default=str), encoding="utf-8")
    checkpoint("model_load", "complete", **MODEL_INFO)
except Exception as exc:
    checkpoint("model_load", "error", error=repr(exc), traceback=traceback.format_exc()[-4000:])
    raise
'''),
    md("""## Inference and batch checkpoints

The full-input pass is the primary benchmark. The question-only pass is an out-of-distribution diagnostic. Every batch is appended to JSONL before moving on; incomplete runs therefore retain all completed batches.
"""),
    code(r'''def make_text(row, question_only=False):
    passage_text = "" if question_only else "\n".join(row.get("passages", []))[:1500]
    return f"Sub-question: {row['question']}\n\nPassages:\n{passage_text}"

def run_view(view, question_only=False):
    output_path = RUN_DIR / f"predictions_{view}.jsonl"
    complete = True
    processed = 0
    batches = 0
    with output_path.open("w", encoding="utf-8") as sink:
        for dataset, group in DF.groupby("dataset", sort=True):
            records = group.to_dict("records")
            for start in range(0, len(records), BATCH_SIZE):
                if elapsed_seconds() >= SOFT_STOP_SECONDS or remaining_seconds() <= SAFETY_MARGIN_SECONDS:
                    complete = False
                    checkpoint(view, "partial", dataset=dataset, processed=processed, remaining_seconds=round(remaining_seconds(), 1))
                    break
                batch = records[start:start + BATCH_SIZE]
                texts = [make_text(row, question_only=question_only) for row in batch]
                encoded = TOKENIZER(
                    texts, max_length=MAX_LENGTH, padding="max_length",
                    truncation=True, return_tensors="pt",
                )
                token_lengths = encoded["attention_mask"].sum(dim=1).tolist()
                encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
                batch_start = time.monotonic()
                with torch.inference_mode():
                    logits = MODEL(**encoded).logits.float()
                    probabilities = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist()
                batch_seconds = time.monotonic() - batch_start
                batch_id = f"{view}:{dataset}:{start // BATCH_SIZE}"
                for row, probability, token_count in zip(batch, probabilities, token_lengths):
                    prediction = int(probability >= THRESHOLD)
                    record = {
                        **row,
                        "view": view,
                        "prob_answerable": float(probability),
                        "prediction": prediction,
                        "suggested_action": "CONTINUE" if prediction == 1 else "RETRIEVE_MORE",
                        "correct": int(prediction == int(row["label"])),
                        "input_chars": len(make_text(row, question_only=question_only)),
                        "token_count": int(token_count),
                        "batch_id": batch_id,
                        "batch_inference_seconds": round(batch_seconds, 6),
                    }
                    sink.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                sink.flush()
                processed += len(batch)
                batches += 1
                checkpoint(
                    view, "progress", dataset=dataset, processed=processed,
                    total=len(DF), batches=batches,
                    batch_seconds=round(batch_seconds, 3),
                    remaining_seconds=round(remaining_seconds(), 1),
                    saved_predictions=str(output_path),
                )
            if not complete:
                break
    checkpoint(view, "complete" if complete else "partial", processed=processed, total=len(DF), batches=batches, output=str(output_path))
    return complete

try:
    FULL_COMPLETE = run_view("full_input", question_only=False)
    if FULL_COMPLETE and remaining_seconds() > 120:
        QUESTION_ONLY_COMPLETE = run_view("question_only", question_only=True)
    else:
        QUESTION_ONLY_COMPLETE = False
        checkpoint("question_only", "skipped", reason="primary pass incomplete or less than 120 seconds remain")
    INFERENCE_ERROR = None
except Exception as exc:
    INFERENCE_ERROR = repr(exc)
    FULL_COMPLETE = False
    QUESTION_ONLY_COMPLETE = False
    checkpoint("inference", "error", error=repr(exc), traceback=traceback.format_exc()[-4000:])
'''),
    code(r'''def expected_calibration_error(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    ece = 0.0
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (p >= low) & ((p < high) if high < 1 else (p <= high))
        if not mask.any():
            rows.append({"bin": index, "low": low, "high": high, "n": 0, "mean_probability": None, "positive_rate": None})
            continue
        confidence = float(p[mask].mean())
        positive_rate = float(y[mask].mean())
        ece += float(mask.mean()) * abs(positive_rate - confidence)
        rows.append({"bin": index, "low": low, "high": high, "n": int(mask.sum()), "mean_probability": confidence, "positive_rate": positive_rate})
    return float(ece), rows

def analyze_predictions(view):
    path = RUN_DIR / f"predictions_{view}.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        return None
    pred = pd.read_json(path, lines=True)
    if pred.empty:
        return None
    pred["label"] = pred["label"].astype(int)
    pred["prediction"] = pred["prediction"].astype(int)
    pred.to_csv(RUN_DIR / f"predictions_{view}.csv", index=False)
    metrics = []
    calibration = []
    for dataset, group in list(pred.groupby("dataset", sort=True)) + [("ALL", pred)]:
        y = group["label"].to_numpy(dtype=int)
        p = group["prob_answerable"].to_numpy(dtype=float)
        yh = group["prediction"].to_numpy(dtype=int)
        ece, bins = expected_calibration_error(y, p)
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
        batch_time = float(group.drop_duplicates("batch_id")["batch_inference_seconds"].sum())
        metrics.append({
            "view": view, "dataset": dataset, "n": len(group),
            "n_positive": int(y.sum()), "n_negative": int((1 - y).sum()),
            "accuracy": float(accuracy_score(y, yh)),
            "precision": float(precision_score(y, yh, zero_division=0)),
            "recall": float(recall_score(y, yh, zero_division=0)),
            "f1": float(f1_score(y, yh, zero_division=0)),
            "macro_f1": float(f1_score(y, yh, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, yh)) if len(np.unique(y)) == 2 else None,
            "auroc": auc, "ece_15": ece, "brier_score": float(np.mean((p - y) ** 2)),
            "tn_fp_fn_tp": confusion_matrix(y, yh, labels=[0, 1]).ravel().tolist(),
            "inference_seconds": batch_time,
            "latency_ms_per_example": batch_time * 1000 / len(group),
            "status": "evaluated",
        })
        for row in bins:
            calibration.append({"view": view, "dataset": dataset, **row})
    pd.DataFrame(metrics).to_csv(RUN_DIR / f"metrics_{view}.csv", index=False)
    pd.DataFrame(calibration).to_csv(RUN_DIR / f"calibration_{view}.csv", index=False)
    errors = pred[pred["correct"] == 0].copy()
    errors["error_type"] = np.where(
        (errors["label"] == 0) & (errors["prediction"] == 1),
        "false_positive_missing_evidence", "false_negative_sufficient_evidence",
    )
    errors.to_csv(RUN_DIR / f"errors_{view}.csv", index=False)
    return pred, metrics

FULL_ANALYSIS = None
QUESTION_ONLY_ANALYSIS = None
ANALYSIS_ERROR = None
try:
    FULL_ANALYSIS = analyze_predictions("full_input")
    QUESTION_ONLY_ANALYSIS = analyze_predictions("question_only") if QUESTION_ONLY_COMPLETE else None
    if FULL_ANALYSIS is not None:
        full_pred = FULL_ANALYSIS[0]
        eda = full_pred.groupby(["dataset", "label"], dropna=False).agg(
            n=("id", "count"), mean_input_chars=("input_chars", "mean"),
            p50_input_chars=("input_chars", "median"), mean_token_count=("token_count", "mean"),
            p95_token_count=("token_count", lambda s: float(np.percentile(s, 95))),
            mean_probability=("prob_answerable", "mean"), error_rate=("correct", lambda s: 1 - float(s.mean())),
        ).reset_index()
        eda.to_csv(RUN_DIR / "eda_by_dataset_label.csv", index=False)
    checkpoint("analysis", "complete", full_input_rows=0 if FULL_ANALYSIS is None else len(FULL_ANALYSIS[0]), question_only_rows=0 if QUESTION_ONLY_ANALYSIS is None else len(QUESTION_ONLY_ANALYSIS[0]))
except Exception as exc:
    ANALYSIS_ERROR = repr(exc)
    checkpoint("analysis", "error", error=repr(exc), traceback=traceback.format_exc()[-4000:])
'''),
    md("""## Saved artifacts

The final cell writes a run summary and a zip bundle. `progress.jsonl` plus `run_state.json` record the stage and batch checkpoints. Prediction JSONL files are the incremental recovery files; CSV files are the EDA-friendly copies.
"""),
    code(r'''try:
    summary = {
        "run_id": RUN_ID,
        "started_at_utc": RUN_STARTED_UTC,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed_seconds(), 3),
        "budget_seconds": MAX_RUNTIME_SECONDS,
        "soft_stop_seconds": SOFT_STOP_SECONDS,
        "status": "complete" if FULL_COMPLETE and QUESTION_ONLY_COMPLETE else ("primary_complete_question_only_skipped" if FULL_COMPLETE else "partial"),
        "scope": "classifier inference and question-only diagnostic; no generator/retriever/training",
        "n_prepared_examples": len(DF),
        "counts": DF.groupby("dataset").size().to_dict(),
        "full_input_complete": bool(FULL_COMPLETE),
        "question_only_complete": bool(QUESTION_ONLY_COMPLETE),
        "inference_error": INFERENCE_ERROR,
        "analysis_error": ANALYSIS_ERROR,
        "environment": ENVIRONMENT,
        "model": MODEL_INFO,
        "checkpoint_source": str(CHECKPOINT_DIR),
        "evaluation_manifest": EVAL_MANIFEST,
        "metrics_full_input": [] if FULL_ANALYSIS is None else FULL_ANALYSIS[1],
        "metrics_question_only": [] if QUESTION_ONLY_ANALYSIS is None else QUESTION_ONLY_ANALYSIS[1],
        "not_evaluated": EVAL_MANIFEST.get("evaluation_data", {}).get("not_evaluated", {}),
    }
    (RUN_DIR / "run_metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (RUN_DIR / "README.md").write_text(
        "# 15-minute classifier EDA run\n\n"
        "This run evaluates the saved answerability classifier on prepared labeled examples. It does not run a generator or retrieval system, so it is not an end-to-end reproduction of the paper's Table 7.\n\n"
        "Key files: `metrics_full_input.csv`, `predictions_full_input.csv`, `errors_full_input.csv`, `calibration_full_input.csv`, `eda_by_dataset_label.csv`, `progress.jsonl`, and `run_metadata.json`.\n",
        encoding="utf-8",
    )
    artifact_inventory = []
    for path in sorted(RUN_DIR.rglob("*")):
        if path.is_file() and path.name != "artifact_inventory.csv":
            artifact_inventory.append({"path": str(path.relative_to(RUN_DIR)), "bytes": path.stat().st_size})
    pd.DataFrame(artifact_inventory).to_csv(RUN_DIR / "artifact_inventory.csv", index=False)
    bundle = RUN_DIR.parent / f"fact_grounding_eda_{RUN_ID}.zip"
    shutil.make_archive(str(bundle.with_suffix("")), "zip", root_dir=RUN_DIR)
    checkpoint("finalize", "complete", elapsed_seconds=round(elapsed_seconds(), 3), run_dir=str(RUN_DIR), bundle=str(bundle), artifacts=len(artifact_inventory))
    print("EDA_BUNDLE=", bundle, flush=True)
except Exception as exc:
    checkpoint("finalize", "error", error=repr(exc), traceback=traceback.format_exc()[-4000:])
    raise
'''),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

KERNEL_DIR.mkdir(parents=True, exist_ok=True)
(KERNEL_DIR / "15min_eda_pipeline.ipynb").write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
)
metadata = {
    "id": "nguynbquangminh/fact-grounding-15min-eda",
    "title": "Fact Grounding 15min EDA",
    "code_file": "15min_eda_pipeline.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "enable_internet": "true",
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [
        "nguynbquangminh/fact-grounding-checkpoint",
        "nguynbquangminh/fact-grounding-eval-data",
    ],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
(KERNEL_DIR / "kernel-metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(KERNEL_DIR)
