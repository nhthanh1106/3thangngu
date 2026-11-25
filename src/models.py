"""Hugging Face checkpoint inspection, bounded download, and embedding helpers."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np


KPR_MODEL_ID = "knowledgeable-ai/kpr-bge-base-en-v1.5"
BASELINE_MODEL_ID = "BAAI/bge-base-en-v1.5"
KPR_SOURCE_FILES = ("configuration.py", "modeling.py", "tokenization_kpr.py")


def inspect_hub_model(model_id: str) -> dict[str, Any]:
    """Read repository revision and file sizes from the Hugging Face Hub API."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(model_id, files_metadata=True)
    files = []
    unknown_files = []
    for sibling in info.siblings:
        size = getattr(sibling, "size", None)
        if size is None:
            lfs = getattr(sibling, "lfs", None)
            size = getattr(lfs, "size", None) if lfs is not None else None
        if size is None:
            unknown_files.append(sibling.rfilename)
        else:
            files.append({"name": sibling.rfilename, "size_bytes": int(size)})
    return {
        "model_id": model_id,
        "revision": info.sha,
        "total_repository_size_bytes": sum(item["size_bytes"] for item in files),
        "files": files,
        "unknown_size_files": unknown_files,
        "commit_url": f"https://huggingface.co/{model_id}/tree/{info.sha}",
    }


def inspect_kpr_remote_code(model_id: str, revision: str, cache_dir: str | Path) -> dict[str, Any]:
    """Download only small custom-code files and check expected inference hooks.

    These checks happen before the notebook loads the remote implementation with
    trust_remote_code=True. Source is pinned to the inspected Hub revision.
    """
    from huggingface_hub import hf_hub_download

    cache_dir = Path(cache_dir)
    source = {}
    source_hashes = {}
    for filename in KPR_SOURCE_FILES:
        local_path = Path(
            hf_hub_download(
                repo_id=model_id,
                filename=filename,
                revision=revision,
                cache_dir=str(cache_dir),
                etag_timeout=30,
            )
        )
        contents = local_path.read_text(encoding="utf-8")
        source[filename] = contents
        source_hashes[filename] = _sha256_text(contents)

    model_code = source["modeling.py"]
    tokenizer_code = source["tokenization_kpr.py"]
    config_code = source["configuration.py"]
    checks = {
        "entity_fusion_attention_present": "EntityFusionMultiHeadAttention" in model_code,
        "entity_fusion_layer_present": "EntityFusionLayer" in model_code,
        "entity_inputs_consumed": "entity_ids" in model_code and "entity_embeds" in model_code,
        "memory_mapped_entity_embeddings": 'mmap_mode="r"' in tokenizer_code or "mmap_mode='r'" in tokenizer_code,
        "custom_bert_configuration_present": "KPRConfigForBert" in config_code,
        "entity_linking_and_positions_present": (
            "detect_mentions" in tokenizer_code and "entity_position_ids" in tokenizer_code
        ),
        "marisa_trie_dependency_present": "marisa_trie" in tokenizer_code,
        "spacy_dependency_present": "import spacy" in tokenizer_code,
    }
    missing = [name for name, passed in checks.items() if not passed]
    if missing:
        raise RuntimeError(f"KPR custom-code inspection did not pass expected checks: {missing}")

    return {
        "revision": revision,
        "source_files": list(KPR_SOURCE_FILES),
        "source_sha256": source_hashes,
        "checks": checks,
        "inference_observation": (
            "The Hub tokenizer links detected entity mentions to entity IDs and token positions, "
            "reads the large entity table with NumPy memory mapping, and batches selected entity "
            "vectors into entity_embeds. The KPR model encodes CLS text vectors and fuses entity "
            "vectors with context using entity-fusion attention."
        ),
        "source_code_executed_during_inspection": False,
    }


def required_snapshot_files(model_info: dict[str, Any]) -> tuple[list[str], int]:
    """Return inference files and their known byte total, excluding card metadata."""
    excluded = {"README.md", ".gitattributes"}
    required = [item for item in model_info["files"] if item["name"] not in excluded]
    unknown_required = [name for name in model_info["unknown_size_files"] if name not in excluded]
    if unknown_required:
        raise RuntimeError(f"Cannot estimate required checkpoint disk usage; missing sizes for {unknown_required}.")
    if not required:
        raise RuntimeError(f"Hub API returned no sized inference files for {model_info['model_id']}.")
    return [item["name"] for item in required], sum(item["size_bytes"] for item in required)


def cache_size_bytes(path: str | Path) -> int:
    """Sum regular-file sizes without counting symlink targets twice."""
    path = Path(path)
    if not path.exists():
        return 0
    total = 0
    seen: set[str] = set()
    for file_path in path.rglob("*"):
        if not file_path.is_file():
            continue
        try:
            resolved = str(file_path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            total += file_path.stat().st_size
        except OSError:
            continue
    return total


def download_snapshot_with_deadline(
    model_id: str,
    revision: str,
    cache_dir: str | Path,
    allow_patterns: list[str],
    *,
    stop_at_monotonic: float,
    poll_seconds: float = 2.0,
    max_workers: int = 4,
) -> str:
    """Download through a child process so a slow Hub request can be stopped.

    The parent notebook remains able to save its partial JSON report if the soft
    deadline is reached. A process exit caused by network/disk failure is raised
    as an error and never treated as a successful KPR load.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker_path = cache_dir / f"snapshot-path-{uuid.uuid4().hex}.txt"
    download_script = (
        "import json, sys\n"
        "from huggingface_hub import snapshot_download\n"
        "repo_id, revision, cache_dir, patterns_json, marker_path, workers = sys.argv[1:]\n"
        "path = snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir, "
        "allow_patterns=json.loads(patterns_json), max_workers=int(workers), etag_timeout=30)\n"
        "open(marker_path, 'w', encoding='utf-8').write(path)\n"
    )
    process_env = os.environ.copy()
    process_env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "45")
    process_env.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    command = [
        sys.executable,
        "-u",
        "-c",
        download_script,
        model_id,
        revision,
        str(cache_dir),
        json.dumps(allow_patterns),
        str(marker_path),
        str(max_workers),
    ]
    process = subprocess.Popen(command, env=process_env)
    try:
        while process.poll() is None:
            if time.monotonic() >= stop_at_monotonic:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                raise TimeoutError(
                    f"Download of {model_id} did not finish before the notebook soft deadline; "
                    "partial Hugging Face cache files may remain in the disposable cache."
                )
            time.sleep(poll_seconds)
        if process.returncode != 0:
            raise RuntimeError(f"Hugging Face snapshot download failed with child exit code {process.returncode}.")
        if not marker_path.exists():
            raise RuntimeError("Hugging Face downloader exited without returning a snapshot path.")
        snapshot_path = marker_path.read_text(encoding="utf-8").strip()
        if not snapshot_path or not Path(snapshot_path).is_dir():
            raise RuntimeError(f"Hugging Face returned an invalid snapshot path: {snapshot_path!r}")
        return snapshot_path
    finally:
        marker_path.unlink(missing_ok=True)


def load_sentence_transformer(
    model_path: str | Path,
    *,
    device: str,
    trust_remote_code: bool,
    max_seq_length: int = 256,
) -> tuple[Any, dict[str, Any]]:
    """Load a local SentenceTransformer checkpoint and record its load metadata."""
    from sentence_transformers import SentenceTransformer

    started = time.monotonic()
    load_fallback = None
    try:
        model = SentenceTransformer(
            str(model_path),
            trust_remote_code=trust_remote_code,
            device=device,
            local_files_only=True,
        )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or not str(device).startswith("cuda"):
            raise
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        load_fallback = "CUDA out of memory during model loading; reloaded the same checkpoint on CPU."
        model = SentenceTransformer(
            str(model_path),
            trust_remote_code=trust_remote_code,
            device="cpu",
            local_files_only=True,
        )
    model.max_seq_length = max_seq_length
    elapsed = time.monotonic() - started
    parameters = list(model.parameters())
    dtype = str(parameters[0].dtype).replace("torch.", "") if parameters else "unknown"
    return model, {
        "load_seconds": elapsed,
        "embedding_dimension": model.get_sentence_embedding_dimension(),
        "device": str(model.device),
        "requested_device": device,
        "load_fallback": load_fallback,
        "precision": dtype,
        "max_sequence_length": max_seq_length,
    }


def encode_texts_batched(
    model: Any,
    texts: list[str],
    *,
    batch_size: int,
    deadline_monotonic: float,
    description: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode in bounded microbatches, reducing the batch after CUDA OOM."""
    if not texts:
        raise ValueError(f"Cannot encode an empty {description} collection.")
    outputs = []
    position = 0
    effective_batch_size = max(1, int(batch_size))
    oom_retries = 0
    cpu_fallbacks = 0
    started = time.monotonic()
    while position < len(texts):
        if time.monotonic() >= deadline_monotonic:
            raise TimeoutError(f"Soft execution deadline reached while encoding {description}.")
        chunk = texts[position : position + effective_batch_size]
        try:
            embeddings = model.encode(
                chunk,
                batch_size=len(chunk),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            try:
                import torch

                if torch.cuda.is_available() and effective_batch_size > 1:
                    torch.cuda.empty_cache()
                    oom_retries += 1
                    effective_batch_size = max(1, effective_batch_size // 2)
                    continue
            except Exception:
                pass
            if str(getattr(model, "device", "")).startswith("cuda"):
                try:
                    model.to("cpu")
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    cpu_fallbacks += 1
                    oom_retries += 1
                    effective_batch_size = 1
                    continue
                except Exception as fallback_error:
                    raise RuntimeError(
                        f"{description} ran out of GPU memory even at batch size 1, and CPU fallback failed: "
                        f"{type(fallback_error).__name__}: {fallback_error}"
                    ) from exc
            raise
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunk):
            raise ValueError(
                f"{description} embeddings have invalid shape {embeddings.shape}; expected ({len(chunk)}, dim)."
            )
        if not np.isfinite(embeddings).all():
            raise ValueError(f"{description} embeddings contain NaN or infinite values.")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=2e-3):
            raise ValueError(f"{description} embeddings are not L2-normalized as requested.")
        outputs.append(embeddings)
        position += len(chunk)

    result = np.vstack(outputs)
    elapsed = time.monotonic() - started
    return result, {
        "item_count": len(texts),
        "seconds": elapsed,
        "seconds_per_item": elapsed / len(texts),
        "requested_batch_size": int(batch_size),
        "effective_batch_size": effective_batch_size,
        "oom_retries": oom_retries,
        "cpu_fallbacks": cpu_fallbacks,
        "effective_device": str(getattr(model, "device", "unknown")),
        "embedding_dimension": int(result.shape[1]),
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
