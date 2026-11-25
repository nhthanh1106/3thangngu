"""Small, deterministic BEIR SciFact sampling helpers."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def download_scifact_archive(
    cache_dir: str | Path,
    *,
    deadline_monotonic: float | None = None,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Download the small official BEIR SciFact archive to a disposable cache."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / "scifact.zip"
    if archive_path.exists() and archive_path.stat().st_size > 0:
        return {
            "path": str(archive_path),
            "source_url": SCIFACT_URL,
            "bytes": archive_path.stat().st_size,
            "sha256": _sha256_file(archive_path),
            "cache_hit": True,
        }

    temporary_path = archive_path.with_suffix(".zip.part")
    request = urllib.request.Request(SCIFACT_URL, headers={"User-Agent": "kpr-kaggle-retrieval-evaluation/1.0"})
    downloaded = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response, temporary_path.open("wb") as out:
            while True:
                if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                    raise TimeoutError("Soft execution deadline reached while downloading SciFact.")
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
        temporary_path.replace(archive_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return {
        "path": str(archive_path),
        "source_url": SCIFACT_URL,
        "bytes": downloaded,
        "sha256": _sha256_file(archive_path),
        "cache_hit": False,
    }


def load_and_sample_scifact(
    archive_path: str | Path,
    *,
    target_queries: int = 40,
    target_candidates: int = 800,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Load SciFact and form one restricted candidate pool shared by all models.

    Every positive qrel for each selected query is retained. Remaining passages
    are sampled uniformly without replacement from documents not judged
    relevant to any selected query.
    """
    archive_path = Path(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        corpus_member = _find_member(names, "corpus.jsonl")
        queries_member = _find_member(names, "queries.jsonl")
        qrels_member = _find_member(names, "qrels/test.tsv")

        corpus = {
            str(row["_id"]): {
                "title": str(row.get("title", "") or ""),
                "text": str(row.get("text", "") or ""),
            }
            for row in _read_jsonl(archive.read(corpus_member))
        }
        queries = {
            str(row["_id"]): str(row.get("text", "") or "")
            for row in _read_jsonl(archive.read(queries_member))
        }
        qrels: dict[str, dict[str, int]] = {}
        qrels_text = io.StringIO(archive.read(qrels_member).decode("utf-8"))
        for row in csv.DictReader(qrels_text, delimiter="\t"):
            query_id = str(row["query-id"])
            document_id = str(row["corpus-id"])
            grade = int(row["score"])
            if grade > 0 and query_id in queries and document_id in corpus:
                qrels.setdefault(query_id, {})[document_id] = grade

    return sample_retrieval_data(
        corpus,
        queries,
        qrels,
        target_queries=target_queries,
        target_candidates=target_candidates,
        random_seed=random_seed,
        metadata={
            "name": "BEIR SciFact",
            "source_url": SCIFACT_URL,
            "source_kind": "official_beir_archive",
            "qrels_split": "test",
            "archive_sha256": _sha256_file(archive_path),
        },
    )


def load_and_sample_scifact_hf_mirror(
    cache_dir: str | Path,
    *,
    target_queries: int = 40,
    target_candidates: int = 800,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Load the BEIR organization's Hub mirror if its archive URL is unavailable."""
    from huggingface_hub import HfApi, hf_hub_download
    import pandas as pd

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    download_started = time.monotonic()
    api = HfApi()
    corpus_info = api.dataset_info("BeIR/scifact", files_metadata=True)
    qrels_info = api.dataset_info("BeIR/scifact-qrels", files_metadata=True)

    corpus_names = [item.rfilename for item in corpus_info.siblings]
    query_names = corpus_names
    corpus_files = sorted(name for name in corpus_names if name.startswith("corpus/") and name.endswith(".parquet"))
    query_files = sorted(name for name in query_names if name.startswith("queries/") and name.endswith(".parquet"))
    qrels_names = [item.rfilename for item in qrels_info.siblings]
    if not corpus_files or not query_files or "test.tsv" not in qrels_names:
        raise ValueError("Official BEIR Hub mirror is missing corpus, queries, or test qrels files.")

    corpus_paths = [
        Path(
            hf_hub_download(
                repo_id="BeIR/scifact",
                repo_type="dataset",
                filename=name,
                revision=corpus_info.sha,
                cache_dir=str(cache_dir),
                etag_timeout=30,
            )
        )
        for name in corpus_files
    ]
    query_paths = [
        Path(
            hf_hub_download(
                repo_id="BeIR/scifact",
                repo_type="dataset",
                filename=name,
                revision=corpus_info.sha,
                cache_dir=str(cache_dir),
                etag_timeout=30,
            )
        )
        for name in query_files
    ]
    qrels_path = hf_hub_download(
        repo_id="BeIR/scifact-qrels",
        repo_type="dataset",
        filename="test.tsv",
        revision=qrels_info.sha,
        cache_dir=str(cache_dir),
        etag_timeout=30,
    )
    download_seconds = time.monotonic() - download_started
    preparation_started = time.monotonic()
    corpus_df = pd.concat([pd.read_parquet(path) for path in corpus_paths], ignore_index=True)
    query_df = pd.concat([pd.read_parquet(path) for path in query_paths], ignore_index=True)
    qrels_df = pd.read_csv(qrels_path, sep="\t")
    corpus = {
        str(row["_id"]): {"title": str(row.get("title", "") or ""), "text": str(row.get("text", "") or "")}
        for row in corpus_df.to_dict(orient="records")
    }
    queries = {
        str(row["_id"]): str(row.get("text", "") or "") for row in query_df.to_dict(orient="records")
    }
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_df.to_dict(orient="records"):
        query_id = str(row["query-id"])
        document_id = str(row["corpus-id"])
        grade = int(row["score"])
        if grade > 0 and query_id in queries and document_id in corpus:
            qrels.setdefault(query_id, {})[document_id] = grade

    result = sample_retrieval_data(
        corpus,
        queries,
        qrels,
        target_queries=target_queries,
        target_candidates=target_candidates,
        random_seed=random_seed,
        metadata={
            "name": "BEIR SciFact",
            "source_url": "https://huggingface.co/datasets/BeIR/scifact",
            "qrels_source_url": "https://huggingface.co/datasets/BeIR/scifact-qrels",
            "source_kind": "official_beir_huggingface_mirror_with_test_qrels",
            "qrels_split": "test",
            "dataset_revision": corpus_info.sha,
            "qrels_revision": qrels_info.sha,
            "corpus_sha256": _sha256_files(corpus_paths),
            "qrels_sha256": _sha256_file(Path(qrels_path)),
        },
    )
    result["metadata"]["download_seconds"] = download_seconds
    result["metadata"]["preparation_seconds"] = time.monotonic() - preparation_started
    return result


def diagnostic_demo_sample() -> dict[str, Any]:
    """Return a tiny hand-authored diagnostic set, never a real benchmark."""
    queries = {
        "diag-q1": "Which alloy is used to make lightweight aircraft frames?",
        "diag-q2": "What process converts atmospheric nitrogen into ammonia?",
        "diag-q3": "Which cells produce insulin in the pancreas?",
    }
    corpus = {
        "diag-d1": {"title": "Aircraft materials", "text": "Aluminum alloys are widely used in lightweight aircraft frames."},
        "diag-d2": {"title": "Aircraft engines", "text": "Turbofan engines power many commercial aircraft."},
        "diag-d3": {"title": "Ammonia synthesis", "text": "The Haber-Bosch process converts nitrogen and hydrogen into ammonia."},
        "diag-d4": {"title": "Nitrogen cycle", "text": "Microbes transform nitrogen compounds in soil and water."},
        "diag-d5": {"title": "Pancreatic islets", "text": "Beta cells in pancreatic islets produce insulin."},
        "diag-d6": {"title": "Hormone signaling", "text": "Glucagon helps regulate blood glucose levels."},
    }
    qrels = {"diag-q1": {"diag-d1": 1}, "diag-q2": {"diag-d3": 1}, "diag-q3": {"diag-d5": 1}}
    return {
        "query_ids": list(queries),
        "queries": queries,
        "candidate_ids": list(corpus),
        "corpus": corpus,
        "qrels": qrels,
        "metadata": {
            "name": "Hand-authored retrieval diagnostic demo",
            "source_url": None,
            "source_kind": "synthetic_diagnostic_only",
            "qrels_split": None,
            "evaluation_type": "diagnostic_only",
            "contains_real_benchmark_labels": False,
            "selection_procedure": "Fixed 3-query, 6-document diagnostic fixture authored in code; not measured scientific dataset labels.",
            "candidate_pool_scope": "diagnostic_only",
            "relevance_note": "Hand-authored labels exist only to smoke-check the retrieval workflow; do not report these metrics as benchmark results.",
            "random_seed": None,
            "selected_queries": 3,
            "candidate_passages": 6,
            "judged_relevant_passages_in_pool": 3,
            "sampled_distractor_passages": 3,
        },
    }


def sample_retrieval_data(
    corpus: dict[str, dict[str, str]],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    *,
    target_queries: int,
    target_candidates: int,
    random_seed: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    eligible_query_ids = sorted(
        (query_id for query_id, relevant in qrels.items() if relevant), key=_stable_id_key
    )
    if not eligible_query_ids:
        raise ValueError("SciFact test qrels contained no usable query-document relevance labels.")

    rng = random.Random(random_seed)
    rng.shuffle(eligible_query_ids)
    selected_query_ids = eligible_query_ids[: min(target_queries, len(eligible_query_ids))]
    if not selected_query_ids:
        raise ValueError("No labeled SciFact queries were selected.")

    selected_qrels = {query_id: qrels[query_id] for query_id in selected_query_ids}
    relevant_document_ids = sorted(
        {document_id for rels in selected_qrels.values() for document_id in rels}, key=_stable_id_key
    )
    if len(relevant_document_ids) > target_candidates:
        raise ValueError(
            f"Selected queries require {len(relevant_document_ids)} judged relevant passages, "
            f"exceeding the target candidate pool of {target_candidates}."
        )

    candidate_count = min(max(target_candidates, len(relevant_document_ids)), len(corpus))
    distractor_pool = sorted(set(corpus) - set(relevant_document_ids), key=_stable_id_key)
    distractor_count = candidate_count - len(relevant_document_ids)
    distractor_ids = rng.sample(distractor_pool, distractor_count)
    candidate_ids = sorted(relevant_document_ids + distractor_ids, key=_stable_id_key)

    selected_queries = {query_id: queries[query_id] for query_id in selected_query_ids}
    selected_corpus = {document_id: corpus[document_id] for document_id in candidate_ids}
    if any(not (set(rels) <= set(selected_corpus)) for rels in selected_qrels.values()):
        raise AssertionError("The sampled candidate pool is missing a judged relevant passage.")

    full_metadata = {
            **metadata,
            "random_seed": random_seed,
            "eligible_labeled_queries": len(eligible_query_ids),
            "selected_queries": len(selected_query_ids),
            "candidate_passages": len(candidate_ids),
            "judged_relevant_passages_in_pool": len(relevant_document_ids),
            "sampled_distractor_passages": len(distractor_ids),
            "selection_procedure": (
                "Sort test-qrel query IDs, shuffle with Python random.Random(seed), take the first "
                "target_queries; include every positive-qrel passage for those queries; sample the "
                "remaining candidate passages uniformly without replacement from other corpus IDs; "
                "sort final IDs for deterministic tie-breaking."
            ),
            "candidate_pool_scope": "restricted_candidate_pool_not_full_corpus_BEIR",
            "relevance_note": (
                "SciFact qrels encode retrieval relevance for claim-evidence retrieval. They do not "
                "label whether every retrieved passage supports or contradicts a claim."
            ),
        }
    return {
        "query_ids": selected_query_ids,
        "queries": selected_queries,
        "candidate_ids": candidate_ids,
        "corpus": selected_corpus,
        "qrels": selected_qrels,
        "metadata": full_metadata,
    }


def passage_text(document: dict[str, str]) -> str:
    """Format the BEIR title and abstract as a single passage input."""
    title = document.get("title", "").strip()
    body = document.get("text", "").strip()
    if title and body:
        return f"{title}\n{body}"
    return title or body


def _find_member(names: list[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one archive member ending in {suffix!r}; found {matches!r}.")
    return matches[0]


def _read_jsonl(raw: bytes):
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if line.strip():
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record at line {line_number}: {exc}") from exc


def _stable_id_key(value: str):
    return (0, int(value)) if value.isdigit() else (1, value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_files(paths: list[Path]) -> dict[str, str]:
    return {str(path.name): _sha256_file(path) for path in paths}
