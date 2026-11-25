"""Scoring, metric calculations, and paired query-level comparisons."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


METRIC_NAMES = ("recall_at_1", "recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10")


def evaluate_retrieval(
    query_embeddings: np.ndarray,
    passage_embeddings: np.ndarray,
    query_ids: list[str],
    candidate_ids: list[str],
    qrels: dict[str, dict[str, int]],
    corpus: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Compute cosine-score rankings and standard BEIR-style top-k metrics."""
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    passage_embeddings = np.asarray(passage_embeddings, dtype=np.float32)
    if query_embeddings.ndim != 2 or passage_embeddings.ndim != 2:
        raise ValueError("Query and passage embeddings must both be 2D arrays.")
    if query_embeddings.shape[0] != len(query_ids):
        raise ValueError("Query embedding count does not match query IDs.")
    if passage_embeddings.shape[0] != len(candidate_ids):
        raise ValueError("Passage embedding count does not match candidate IDs.")
    if query_embeddings.shape[1] != passage_embeddings.shape[1]:
        raise ValueError(
            f"Embedding dimensions differ: query={query_embeddings.shape[1]}, passage={passage_embeddings.shape[1]}."
        )
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Candidate document IDs must be unique.")
    if any(query_id not in qrels or not qrels[query_id] for query_id in query_ids):
        raise ValueError("Every evaluated query must have at least one positive qrel.")
    if any(not set(qrels[query_id]) <= set(candidate_ids) for query_id in query_ids):
        raise ValueError("The candidate pool must include all judged relevant documents for each query.")
    if not np.isfinite(query_embeddings).all() or not np.isfinite(passage_embeddings).all():
        raise ValueError("Embedding matrices contain NaN or infinite values.")

    scoring_started = time.monotonic()
    similarities = query_embeddings @ passage_embeddings.T
    scoring_seconds = time.monotonic() - scoring_started
    if similarities.shape != (len(query_ids), len(candidate_ids)) or not np.isfinite(similarities).all():
        raise ValueError("Similarity matrix has an invalid shape or non-finite values.")

    per_query = []
    metric_totals = {name: 0.0 for name in METRIC_NAMES}
    for row_index, query_id in enumerate(query_ids):
        order = np.argsort(-similarities[row_index], kind="stable")
        ranked_ids = [candidate_ids[index] for index in order]
        relevance = qrels[query_id]
        query_metrics = _metrics_for_ranking(ranked_ids, relevance)
        for metric_name in METRIC_NAMES:
            metric_totals[metric_name] += query_metrics[metric_name]

        top_10 = []
        for rank, candidate_index in enumerate(order[:10], start=1):
            document_id = candidate_ids[candidate_index]
            document = corpus[document_id]
            top_10.append(
                {
                    "document_id": document_id,
                    "rank": rank,
                    "score": float(similarities[row_index, candidate_index]),
                    "is_relevant": document_id in relevance,
                    "relevance_grade": relevance.get(document_id, 0),
                    "passage_text": _passage_text(document),
                }
            )
        per_query.append(
            {
                "query_id": query_id,
                "metrics": query_metrics,
                "gold_relevant_document_ids": sorted(relevance, key=_stable_id_key),
                "top_10": top_10,
            }
        )

    count = len(query_ids)
    return {
        "metrics": {name: metric_totals[name] / count for name in METRIC_NAMES},
        "per_query": per_query,
        "scoring_seconds": scoring_seconds,
        "scored_pairs": len(query_ids) * len(candidate_ids),
        "similarity_function": "dot_product_of_l2_normalized_embeddings (cosine similarity)",
    }


def combine_query_results(
    query_ids: list[str],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    baseline_result: dict[str, Any],
    kpr_result: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create detailed paired predictions and up to five examples per category."""
    baseline_rows = {row["query_id"]: row for row in baseline_result["per_query"]}
    kpr_rows = {row["query_id"]: row for row in kpr_result["per_query"]}
    combined = []
    for query_id in query_ids:
        baseline = baseline_rows[query_id]
        kpr = kpr_rows[query_id]
        baseline_ndcg = baseline["metrics"]["ndcg_at_10"]
        kpr_ndcg = kpr["metrics"]["ndcg_at_10"]
        if kpr_ndcg > baseline_ndcg + 1e-12:
            direction = "improved"
        elif kpr_ndcg < baseline_ndcg - 1e-12:
            direction = "worsened"
        else:
            direction = "tie"
        combined.append(
            {
                "query_id": query_id,
                "query_text": queries[query_id],
                "gold_relevant_document_ids": sorted(qrels[query_id], key=_stable_id_key),
                "baseline": baseline,
                "kpr": kpr,
                "comparison": {
                    "ranking_change_by_ndcg_at_10": direction,
                    "ndcg_at_10_delta_kpr_minus_baseline": kpr_ndcg - baseline_ndcg,
                    "mrr_at_10_delta_kpr_minus_baseline": (
                        kpr["metrics"]["mrr_at_10"] - baseline["metrics"]["mrr_at_10"]
                    ),
                    "recall_at_10_delta_kpr_minus_baseline": (
                        kpr["metrics"]["recall_at_10"] - baseline["metrics"]["recall_at_10"]
                    ),
                },
            }
        )

    kpr_better = [row for row in combined if row["comparison"]["ranking_change_by_ndcg_at_10"] == "improved"]
    baseline_better = [row for row in combined if row["comparison"]["ranking_change_by_ndcg_at_10"] == "worsened"]
    both_failed = [
        row
        for row in combined
        if row["baseline"]["metrics"]["recall_at_10"] == 0.0
        and row["kpr"]["metrics"]["recall_at_10"] == 0.0
    ]
    kpr_better.sort(key=lambda row: row["comparison"]["ndcg_at_10_delta_kpr_minus_baseline"], reverse=True)
    baseline_better.sort(key=lambda row: row["comparison"]["ndcg_at_10_delta_kpr_minus_baseline"])
    both_failed.sort(key=lambda row: _stable_id_key(row["query_id"]))

    def failure_case(row: dict[str, Any], case_type: str) -> dict[str, Any]:
        return {
            "case_type": case_type,
            "query_id": row["query_id"],
            "query_text": row["query_text"],
            "gold_relevant_document_ids": row["gold_relevant_document_ids"],
            "category": "unclassified",
            "tentative_observation": (
                "The paired top-10 rankings and SciFact qrels are included for manual inspection; "
                "no causal failure category is inferred from relevance labels alone."
            ),
            "baseline_top_10": row["baseline"]["top_10"],
            "kpr_top_10": row["kpr"]["top_10"],
            "baseline_metrics": row["baseline"]["metrics"],
            "kpr_metrics": row["kpr"]["metrics"],
        }

    categories = {
        "kpr_better": [failure_case(row, "kpr_better_by_ndcg_at_10") for row in kpr_better[:5]],
        "baseline_better": [failure_case(row, "baseline_better_by_ndcg_at_10") for row in baseline_better[:5]],
        "both_failed": [failure_case(row, "neither_retrieved_relevant_in_top_10") for row in both_failed[:5]],
    }
    counts = {
        "kpr_better_available": len(kpr_better),
        "baseline_better_available": len(baseline_better),
        "both_failed_available": len(both_failed),
        "examples_returned_per_category": 5,
        "selection_metric": "per-query nDCG@10; both_failed means zero Recall@10 for both models",
    }
    return combined, {**categories, "counts": counts}


def metric_deltas(baseline_metrics: dict[str, float], kpr_metrics: dict[str, float]) -> dict[str, float]:
    return {name: float(kpr_metrics[name] - baseline_metrics[name]) for name in METRIC_NAMES}


def _metrics_for_ranking(ranked_ids: list[str], relevance: dict[str, int]) -> dict[str, float]:
    relevant_ids = set(relevance)
    relevant_count = len(relevant_ids)
    if relevant_count == 0:
        raise ValueError("Metrics are undefined for a query without judged relevant documents.")
    metrics = {}
    for k in (1, 5, 10):
        metrics[f"recall_at_{k}"] = len(set(ranked_ids[:k]) & relevant_ids) / relevant_count

    reciprocal_rank = 0.0
    for rank, document_id in enumerate(ranked_ids[:10], start=1):
        if document_id in relevant_ids:
            reciprocal_rank = 1.0 / rank
            break
    metrics["mrr_at_10"] = reciprocal_rank

    dcg = 0.0
    for rank, document_id in enumerate(ranked_ids[:10], start=1):
        grade = relevance.get(document_id, 0)
        if grade > 0:
            dcg += (2**grade - 1) / np.log2(rank + 1)
    ideal_grades = sorted(relevance.values(), reverse=True)[:10]
    ideal_dcg = sum((2**grade - 1) / np.log2(rank + 1) for rank, grade in enumerate(ideal_grades, start=1))
    metrics["ndcg_at_10"] = float(dcg / ideal_dcg) if ideal_dcg > 0 else 0.0
    return {name: float(value) for name, value in metrics.items()}


def _passage_text(document: dict[str, str]) -> str:
    title = document.get("title", "").strip()
    text = document.get("text", "").strip()
    return f"{title}\n{text}" if title and text else title or text


def _stable_id_key(value: str):
    return (0, int(value)) if value.isdigit() else (1, value)
