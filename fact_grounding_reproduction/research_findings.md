# Research Findings

## Status (Updated 2026-11-26)

Training, reload verification, and comprehensive 8-benchmark (9-track) evaluation of the reproduced `microsoft/deberta-v3-large` fact-grounding classifier are **complete**.

- **Canonical checkpoint:** `checkpoint/model.safetensors` (`SHA-256: 93ef195f3840689a70094c8fdbe4c1126a86e9faf1b9382a3040e3d32b470260`), trained on Kaggle `Tesla T4 × 2` (`huthnhnguyn/fact-grounding-training-reproduction-live-logs`) with dynamic padding, `group_by_length=True`, non-reentrant gradient checkpointing, and effective batch size 16.
- **Fresh reload verification:** Passed with `max prob diff = 2.24e-8` (tolerance `1e-3`).
- **8-Benchmark Kaggle run (Version 1):** Succeeded in `1,641.7s` (`27.3 min`) on `Tesla T4 × 2` (`Peak VRAM: 5.15 / 4.61 GiB`). All summary CSV/JSON artifacts are stored in [`results/05_eight_benchmark_eval/`](results/05_eight_benchmark_eval/).

---

## A. In-Domain Reproduction Quality (`01_musique_dev`, 6,404 hops)

| Metric | Paper (EMNLP 2026) | Canonical `93ef195f` | Reference v5 `6bbd13d5` | Delta (`93ef` − Paper) |
|---|---:|---:|---:|---:|
| Precision | 0.805 | **0.8847** | 0.7709 | +0.0797 |
| Recall | 0.765 | 0.7148 | 0.8079 | −0.0502 |
| F1 | 0.785 | **0.7907** | 0.7890 | +0.0057 |
| Accuracy | 0.794* | **0.8142** | 0.7878 | +0.0202* |
| AUROC / PR-AUC | — | **0.8764 / 0.8913** | 0.8755 / 0.8958 | — |
| ECE-15 (Raw → Temp Scaled `T=1.929`) | — | **0.1176 → 0.0372** | 0.1378 | — |

*\*The paper's 0.794 accuracy comes from its full-input Table 5 setting.*

---

## B. Cross-Domain & Multi-Task Generalization (8 Benchmarks / 9 Tracks)

Full analysis and published SOTA comparisons are documented in [`reports/eight_benchmark_sota_comparison.md`](reports/eight_benchmark_sota_comparison.md) and [`reports/bottlenecks_and_literature_gap.md`](reports/bottlenecks_and_literature_gap.md). Key empirical findings:

1. **Format shift dominates domain shift:** On MuSiQue-Full, sub-question hops (`02b`) achieve **AUROC `0.8986` / F1 `0.8375`** (`0.8610` calibrated), while full multi-hop questions (`02a`) drop to **AUROC `0.7755` / F1 `0.6806`** (−12.3 AUROC points). Conversely, cross-domain news (`07_multihop_rag_news`, COLM 2024) achieves **AUROC `0.8911`**, **PR-AUC `0.9850`**, and catches **96.3% (`290/301`)** of `null_query` items.
2. **Incomplete-information multi-hop transfer is strong:** On `03_iirc_dev` (1,301 native questions), zero-shot accuracy reaches **76.48%** (**78.40%** calibrated), **F1 `0.8376`** (**`0.8593`** calibrated), **AUROC `0.7914`**, and calibrated **ECE `0.0776`**.
3. **1,500-char / 512-token truncation drives the worst OOD failure:** `05_2wiki_support_pairs` has `char_trunc_rate = 95.9%`, dragging AUROC down to `0.6012` (and `06_faitheval_pairs` truncates `50.9%` of rows).
4. **Adversarial lexical overlap breaks the fixed `0.5` threshold despite strong ranking:** On `08_squad_v2_dev` and `04_hotpotqa_pairs`, AUROC (`0.8409` / `0.8017`) and Pairwise Accuracy (`88.55%` HotpotQA, `81.83%` FaithEval) remain strong, yet fixed threshold `0.5` triggers heavy false positives (`FP = 4,076` SQuAD2, `2,295` HotpotQA).
5. **Zero-leakage post-hoc Temperature Scaling works across OOD:** Fitting a single scalar temperature `T = 1.9290` strictly on `01_musique_dev` (NLL `0.5272 → 0.4318`) reduces ECE-15 by **20%–68% on 8 of 9 tracks** without accessing target test labels.

---

## C. Limitations & Next Steps

- **Scope boundary:** These results benchmark the standalone evidence-sufficiency classifier. End-to-end multi-hop QA pipeline evaluation (measuring final QA EM/F1, retrieval calls, and latency when pairing the calibrated DeBERTa gate with iterative re-retrieval) remains the next experimental milestone.
- **Duplicate `qid` note in Version 1 (`02a`/`02b`):** Because answerable/unanswerable contrasts share `id` in `musique_full_v1.0_dev.jsonl`, Version 1 scored 1 half per `qid` (`2,417/4,834` balanced rows on `02a`, `6,404/12,808` hops on `02b`). The ID collision is fixed in `05_full_decision_benchmark_eval.ipynb` for any future rerun.
- **Promising v2 direction:** Combine sentence pruning (e.g., *Provence*, ICLR 2025) + adversarial hard-negative fine-tuning (SQuAD 2.0 / FaithEval / MuSiQue-Full) + a calibrated Tier-1/Tier-2 cascade gate (DeBERTa 435M at `14–33 ms/example` handling confident cases, escalating grey-zone cases to an LLM judge).
