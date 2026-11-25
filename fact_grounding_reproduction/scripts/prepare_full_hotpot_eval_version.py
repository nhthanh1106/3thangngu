from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "kaggle_eval_data"
RAW_PATH = ROOT / "datasets" / "raw" / "hotpot_dev_distractor_v1.json"

sys.path.insert(0, str(ROOT / "scripts"))
import prepare_eval_data as preparation  # noqa: E402


def main() -> None:
    hotpot, eligible_questions, source_questions = preparation.hotpot_pairs(
        RAW_PATH, question_count=None
    )
    prepared = STAGING / "prepared_eval_examples.jsonl"
    old_rows = [json.loads(line) for line in prepared.read_text(encoding="utf-8").splitlines() if line.strip()]
    preserved = [row for row in old_rows if row.get("dataset") != "HotpotQA"]
    combined = preserved + hotpot
    tmp = prepared.with_suffix(prepared.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for row in combined:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(prepared)

    manifest_path = STAGING / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    info = manifest.setdefault("evaluation_data", {})
    counts = info.setdefault("sample_counts", {})
    counts["HotpotQA"] = len(hotpot)
    info["hotpotqa_source"] = "Official HotpotQA dev distractor split; local copy was retrieved from the Kaggle mirror and the file format/source fields match the official release."
    info["hotpotqa_license"] = "CC BY-SA 4.0 (official HotpotQA site)"
    info["hotpotqa_source_questions"] = source_questions
    info["hotpotqa_eligible_questions"] = eligible_questions
    info["hotpotqa_construction"] = (
        "All eligible questions from the full official distractor dev split are paired into answerable and evidence-deficient examples. "
        "Positives include all annotated supporting facts (two or more sentences); negatives replace one support fact with a nearest-length non-support sentence. "
        "Manual ambiguity exclusions, answer leakage, invalid support references, and contrasts erased by the 1,500-character input cap are excluded. "
        "Labels are support-fact-derived proxies, not original answerability labels."
    )
    info.setdefault("requested_sample_sizes", {})["HotpotQA"] = "All eligible examples from the 7,405-question dev distractor split; no random sample."
    raw_hash = hashlib.sha256(RAW_PATH.read_bytes()).hexdigest()
    info["hotpotqa_raw_sha256"] = raw_hash
    info["prepared_hotpotqa_examples"] = len(hotpot)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    counts_by_dataset: dict[str, dict[str, int]] = {}
    for row in combined:
        entry = counts_by_dataset.setdefault(row["dataset"], {"label_0": 0, "label_1": 0})
        entry[f"label_{int(row['label'])}"] += 1
    print(json.dumps({
        "source_questions": source_questions,
        "eligible_hotpot_questions": eligible_questions,
        "hotpot_examples": len(hotpot),
        "all_prepared_examples": len(combined),
        "hotpot_raw_sha256": raw_hash,
        "counts": counts_by_dataset,
        "staging": str(STAGING),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
