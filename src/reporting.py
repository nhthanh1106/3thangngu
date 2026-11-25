"""Single-file JSON reporting and environment/resource metadata helpers."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_empty_report() -> dict[str, Any]:
    return {
        "metadata": {
            "project": "Lightweight KPR Retrieval Evaluation",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "paper": "Dynamic Injection of Entity Knowledge into Dense Retrievers (Findings EMNLP 2025)",
            "paper_url": "https://aclanthology.org/2025.findings-emnlp.915/",
            "official_repository": "https://github.com/knowledgeable-embedding/knowledgeable-embedding",
            "kpr_model_card": "https://huggingface.co/knowledgeable-ai/kpr-bge-base-en-v1.5",
            "baseline_model_card": "https://huggingface.co/BAAI/bge-base-en-v1.5",
        },
        "configuration": {
            "random_seed": 42,
            "target_queries": 40,
            "target_candidate_passages": 800,
            "max_sequence_length": 256,
            "batch_size": 8,
            "soft_deadline_minutes": 25,
            "hard_deadline_minutes": 30,
            "precision_policy": "float32 to keep both models comparable and avoid KPR entity-input dtype mismatch",
            "no_training": True,
            "paid_apis": False,
        },
        "environment": {},
        "model_information": {"baseline": {}, "kpr": {}},
        "dataset": {},
        "evaluation_protocol": {
            "evaluation_type": "restricted_candidate_retrieval",
            "metrics": ["Recall@1", "Recall@5", "Recall@10", "MRR@10", "nDCG@10"],
            "metric_definitions": {
                "recall_at_k": "For each query, relevant documents in top-k divided by all positive-qrel documents for that query; macro-average over queries.",
                "mrr_at_10": "Macro-average reciprocal rank of the first positive-qrel document in top 10; zero if none appears.",
                "ndcg_at_10": "Macro-average DCG@10 divided by ideal DCG@10; graded gain is 2^relevance-1 and discount is log2(rank+1).",
            },
            "scoring": "dot product of L2-normalized embeddings (cosine similarity)",
            "query_formatting": {
                "baseline": "BGE recommended retrieval prefix is prepended to queries; passages have no prefix.",
                "kpr": "Raw query and passage text per the KPR SentenceTransformer model-card example.",
            },
            "qrel_interpretation": (
                "Metrics use positive SciFact test qrels as the judged relevant set. Unjudged candidate "
                "passages are treated as non-relevant for this sampled ranking calculation; SciFact "
                "qrels are incomplete and do not establish support versus contradiction for each claim."
            ),
            "candidate_pool_note": "This is a deterministic restricted-candidate evaluation, not full-corpus BEIR or the paper's benchmark.",
            "per_query_ranking_comparison": "KPR versus baseline nDCG@10; ties are recorded explicitly.",
        },
        "metrics": {"baseline": None, "kpr": None, "comparison": None},
        "resource_usage": {},
        "timings": {
            "dependency_install_seconds": None,
            "repository_inspection_seconds": None,
            "dataset_download_seconds": None,
            "dataset_preparation_seconds": None,
            "model_download_seconds": {"baseline": None, "kpr": None},
            "model_load_seconds": {"baseline": None, "kpr": None},
            "warmup_seconds": {"baseline": None, "kpr": None},
            "query_encoding_seconds": {"baseline": None, "kpr": None},
            "passage_encoding_seconds": {"baseline": None, "kpr": None},
            "query_encoding_seconds_per_query": {"baseline": None, "kpr": None},
            "scoring_seconds": {"baseline": None, "kpr": None},
            "evaluation_seconds": None,
            "report_export_seconds": None,
            "elapsed_seconds": 0.0,
        },
        "per_query_results": [],
        "failure_analysis": {"kpr_better": [], "baseline_better": [], "both_failed": [], "counts": {}},
        "research_summary": {
            "observations": [],
            "limitations": [
                "No result from this sampled restricted candidate set should be treated as a full BEIR or paper reproduction.",
                "SciFact retrieval qrels do not label factual support or contradiction for every retrieved passage.",
            ],
            "potential_research_questions": [
                "Does the effect persist on the full SciFact corpus and other entity-heavy retrieval datasets?",
                "Which query types benefit from entity injection when judgments are complete and stratified?",
                "How do memory-mapped entity lookup and entity-linking CPU costs affect production latency?",
            ],
            "interpretation_vi": "Chưa có kết quả đo để kết luận KPR tốt hơn BGE. Đây là đánh giá sampled trên candidate pool giới hạn, không phải benchmark đầy đủ.",
        },
        "execution": {
            "status": "failed",
            "completed_steps": [],
            "errors": [],
            "warnings": [],
            "soft_deadline_reached": False,
            "hard_deadline_seconds": 1800,
        },
    }


def save_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Atomically write and parse-verify the one main JSON report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary_path.replace(output_path)
    with output_path.open("r", encoding="utf-8") as stream:
        json.load(stream)
    return output_path


def record_exception(report: dict[str, Any], stage: str, error: BaseException) -> None:
    trace_lines = traceback.format_exception(type(error), error, error.__traceback__)
    report["execution"]["errors"].append(
        {
            "stage": stage,
            "type": type(error).__name__,
            "message": str(error),
            "traceback_summary": "".join(trace_lines[-8:])[-5000:],
        }
    )


def collect_environment(*, include_torch: bool = True) -> dict[str, Any]:
    environment = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "working_directory": os.getcwd(),
        "package_versions": {},
    }
    package_names = (
        "sentence-transformers",
        "transformers",
        "huggingface-hub",
        "safetensors",
        "numpy",
        "requests",
        "psutil",
            "pyarrow",
            "pandas",
            "spacy",
            "marisa-trie",
    ) + (("torch",) if include_torch else ())
    for package in package_names:
        try:
            environment["package_versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            environment["package_versions"][package] = None

    try:
        import psutil

        memory = psutil.virtual_memory()
        environment["system_memory_bytes"] = {"total": memory.total, "available": memory.available}
        environment["process_rss_bytes"] = psutil.Process().memory_info().rss
    except Exception as exc:
        environment["system_memory_bytes"] = None
        environment["process_rss_bytes"] = None
        environment["memory_probe_error"] = f"{type(exc).__name__}: {exc}"

    if include_torch:
        try:
            import torch

            environment["cuda_available"] = bool(torch.cuda.is_available())
            environment["cuda_device_count"] = torch.cuda.device_count()
            environment["cuda_devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            environment["torch_version"] = torch.__version__
            if torch.cuda.is_available():
                environment["cuda_version"] = torch.version.cuda
                environment["gpu_memory_bytes"] = {
                    "total": torch.cuda.get_device_properties(0).total_memory,
                    "free": torch.cuda.mem_get_info(0)[0],
                }
            else:
                environment["gpu_memory_bytes"] = None
        except Exception as exc:
            environment["cuda_available"] = False
            environment["gpu_probe_error"] = f"{type(exc).__name__}: {exc}"
    else:
        environment["cuda_available"] = None
        environment["cuda_device_count"] = None
        environment["gpu_memory_bytes"] = None
    return environment


def collect_runtime_resources(device: str | None = None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"device": device}
    try:
        import psutil

        vm = psutil.virtual_memory()
        snapshot["ram_bytes"] = {"total": vm.total, "available": vm.available}
        snapshot["process_rss_bytes"] = psutil.Process().memory_info().rss
    except Exception as exc:
        snapshot["ram_probe_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import torch

        if torch.cuda.is_available():
            snapshot["gpu_memory_bytes"] = {
                "total": torch.cuda.get_device_properties(0).total_memory,
                "free": torch.cuda.mem_get_info(0)[0],
                "allocated": torch.cuda.memory_allocated(0),
                "reserved": torch.cuda.memory_reserved(0),
                "peak_allocated": torch.cuda.max_memory_allocated(0),
                "peak_reserved": torch.cuda.max_memory_reserved(0),
            }
        else:
            snapshot["gpu_memory_bytes"] = None
    except Exception as exc:
        snapshot["gpu_probe_error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def update_research_summary(report: dict[str, Any]) -> None:
    baseline = report.get("metrics", {}).get("baseline")
    kpr = report.get("metrics", {}).get("kpr")
    comparison = report.get("metrics", {}).get("comparison")
    if not isinstance(baseline, dict) or not isinstance(kpr, dict) or not isinstance(comparison, dict):
        return
    deltas = comparison.get("delta_kpr_minus_baseline", {})
    if report.get("dataset", {}).get("evaluation_type") == "diagnostic_only":
        report["research_summary"]["observations"] = [
            "Chỉ chạy hand-authored diagnostic fixture vì không tải được nguồn SciFact có nhãn; các số đo sau đây chỉ kiểm tra luồng inference và metric.",
            f"Fixture diagnostic có {report.get('dataset', {}).get('selected_query_count', 0)} query và "
            f"{report.get('dataset', {}).get('candidate_passage_count', 0)} passage.",
            "Không diễn giải các metric này như benchmark SciFact hoặc bằng chứng về ưu thế của KPR.",
        ]
        report["research_summary"]["interpretation_vi"] = (
            "Notebook chỉ chạy diagnostic_only bằng dữ liệu minh họa tự tạo vì không tải được dataset có nhãn. "
            "Các metric không phải kết quả benchmark SciFact và không cho phép kết luận KPR tốt hơn BGE."
        )
        return
    observations = [
        (
            f"Trên {report.get('dataset', {}).get('selected_query_count', 'N')} query và cùng "
            f"{report.get('dataset', {}).get('candidate_passage_count', 'N')} candidate passages, "
            f"Recall@10 của KPR={kpr['recall_at_10']:.4f}, BGE={baseline['recall_at_10']:.4f} "
            f"(delta KPR−BGE={deltas.get('recall_at_10', 0.0):+.4f})."
        ),
        (
            f"nDCG@10 của KPR={kpr['ndcg_at_10']:.4f}, BGE={baseline['ndcg_at_10']:.4f}; "
            "đây chỉ là kết quả trên sample và restricted candidate pool này."
        ),
        "SciFact qrels là relevance judgments; chúng không chứng minh passage hỗ trợ hay mâu thuẫn từng claim.",
    ]
    report["research_summary"]["observations"] = observations
    report["research_summary"]["interpretation_vi"] = (
        f"Trong mẫu này, Recall@10 của KPR là {kpr['recall_at_10']:.4f} và của BGE là "
        f"{baseline['recall_at_10']:.4f}; nDCG@10 lần lượt là {kpr['ndcg_at_10']:.4f} và "
        f"{baseline['ndcg_at_10']:.4f}. Đây là kết quả sampled trên restricted candidate pool; "
        "chưa đủ cơ sở để kết luận KPR ưu việt nói chung hoặc suy ra passage hỗ trợ/mâu thuẫn claim."
    )
    if report.get("timings", {}).get("query_encoding_seconds_per_query"):
        report["research_summary"]["observations"].append(
            "Thời gian model loading, query encoding, passage encoding và scoring được ghi riêng; "
            "query latency không gồm download hay model loading."
        )


def finish_execution(report: dict[str, Any], *, both_models_evaluated: bool) -> None:
    has_errors = bool(report["execution"].get("errors"))
    if both_models_evaluated and not has_errors:
        status = "success"
    elif report.get("dataset", {}).get("selected_query_count") or any(
        isinstance(value, dict) for value in report.get("metrics", {}).values()
    ):
        status = "partial_success"
    else:
        status = "failed"
    report["execution"]["status"] = status

