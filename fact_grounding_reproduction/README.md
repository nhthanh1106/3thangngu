# Fact-Grounding Classifier Reproduction & 8-Benchmark Evaluation

This project reproduces the **DeBERTa-v3-large evidence-sufficiency decision classifier** from [Diagnosing the Fact-Grounding Gap in Multi-Hop Question Answering](https://aclanthology.org/2026.emnlp-main.834/) (EMNLP 2026), verifies checkpoint persistence via Git LFS, and evaluates the canonical checkpoint across **8 benchmarks (9 tracks)** spanning in-domain multi-hop hops, native unanswerable multi-hop QA, incomplete-information QA, cross-domain news, faithfulness contrasts, and single-hop answerability control.

## Canonical Checkpoint

- **Location:** [`checkpoint/`](checkpoint/) (Git LFS tracked — see [`checkpoint/README.md`](checkpoint/README.md))
- **File:** `checkpoint/model.safetensors` (`1,740,304,440` bytes)
- **Verified SHA-256:** `93ef195f3840689a70094c8fdbe4c1126a86e9faf1b9382a3040e3d32b470260`
- **Labels:** `0 = NOT-ANSWERABLE`, `1 = ANSWERABLE`
- **Training run:** [`huthnhnguyn/fact-grounding-training-reproduction-live-logs`](https://www.kaggle.com/code/huthnhnguyn/fact-grounding-training-reproduction-live-logs) (`microsoft/deberta-v3-large`, max length 512, passage cap 1,500 chars, LR `2e-5`, effective batch 16, seed 42, Tesla T4 × 2, Transformers 4.57.1)
- **Reload check:** Passed (`max prob diff = 2.24e-8` vs tolerance `1e-3`).

*(Note: An earlier reference checkpoint from `nguynbquangminh/fact-grounding-checkpoint`, SHA-256 `6bbd13d5...`, was evaluated in Notebook 04; its summary metrics are preserved in [`results/04_five_benchmark_v5/`](results/04_five_benchmark_v5/) for comparison. The canonical `93ef195f...` checkpoint outperforms it on in-domain accuracy/precision, pairwise accuracy, and false-positive control.)*

---

## Verified In-Domain Reproduction (`01_musique_dev`, 6,404 hops)

| Metric | Paper (EMNLP 2026) | Canonical `93ef195f` | Difference |
|---|---:|---:|---:|
| Precision | 0.805 | **0.8847** | +0.0797 |
| Recall | 0.765 | 0.7148 | −0.0502 |
| F1 (Answerable) | 0.785 | **0.7907** | +0.0057 |
| Accuracy | 0.794* | **0.8142** | +0.0202* |
| AUROC / PR-AUC | — | **0.8764 / 0.8913** | — |
| ECE-15 (Raw → Temp Scaled `T=1.929`) | — | **0.1176 → 0.0372** | −68.4% |

---

## 8-Benchmark (9-Track) Evaluation Summary (Notebook 05 — Kaggle T4 × 2, 27.3 min)

Executed on Kaggle: [Fact Grounding 8-Benchmark Decision Eval (Version 1)](https://www.kaggle.com/code/huthnhnguyn/fact-grounding-8-benchmark-decision-eval). Full machine-readable outputs are in [`results/05_eight_benchmark_eval/`](results/05_eight_benchmark_eval/).

| Track | Benchmark | `n_scored` | Acc (`0.5`) | F1 (`0.5`) | AUROC | PR-AUC | Pairwise Acc | ECE Raw → Cal (`T=1.929`) | Trunc% | Latency |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `01` | MuSiQue dev (hops) | 6,404 | 0.8142 | 0.7907 | 0.8764 | 0.8913 | — | 0.1176 → **0.0372** | 0.0% | 26.4 ms |
| `02a` | MuSiQue-Full (full-q)\* | 2,417 | 0.6967 | 0.6806 | 0.7755 | 0.7479 | — | 0.1953 → **0.0853** | 50.8% | 29.1 ms |
| `02b` | MuSiQue-Full (sub-q hops)\* | 6,404 | 0.7943 | 0.8375 | **0.8986** | **0.9570** | — | 0.1547 → 0.1482 | 45.0% | 27.2 ms |
| `03` | IIRC dev (EMNLP'20) | 1,301 | 0.7648 | 0.8376 | 0.7914 | 0.8976 | — | 0.1403 → **0.0776** | 25.7% | 25.7 ms |
| `04` | HotpotQA pairs | 8,018 | 0.6737 | 0.7382 | 0.8017 | 0.7881 | **0.8855** | 0.2651 → **0.1957** | 0.0% | 13.8 ms |
| `05` | 2WikiMultiHopQA pairs | 7,188 | 0.5716 | 0.5141 | 0.6012 | 0.5874 | 0.7205 | 0.3287 → **0.2165** | **95.9%** | 32.8 ms |
| `06` | FaithEval Unanswerable (ICLR'25) | 4,982 | 0.6345 | 0.6922 | 0.7215 | 0.7165 | **0.8183** | 0.2632 → **0.1688** | 50.9% | 27.9 ms |
| `07` | MultiHop-RAG News (COLM'24) | 2,556 | 0.7148 | 0.8083 | **0.8911** | **0.9850** | — | 0.2794 → 0.3055 | 11.8% | 26.7 ms |
| `08` | SQuAD 2.0 dev (single-hop ctrl) | 11,873 | 0.6377 | 0.7262 | 0.8409 | 0.8281 | — | 0.3150 → **0.2528** | 4.3% | 16.6 ms |

*\*In Version 1, `02a`/`02b` kept 1 record per unique `qid` (2,417 balanced rows / 6,404 hops) because positive/negative pairs share `id` in `musique_full_v1.0_dev.jsonl`. Fixed in `05_full_decision_benchmark_eval.ipynb` and `scripts/build_full_decision_benchmark_notebook.py` for future reruns.*

---

## Notebooks, Scripts & Reports

### Notebooks
| File | Purpose | Status |
|---|---|---|
| [`01_train_fact_grounding_live.ipynb`](01_train_fact_grounding_live.ipynb) | Fine-tune DeBERTa-v3-large on MuSiQue hops with live logs & recovery checkpoints | Completed (`checkpoint/` persisted via LFS) |
| [`03_verify_persisted_checkpoint.ipynb`](03_verify_persisted_checkpoint.ipynb) | Reload saved checkpoint & verify full 6,404 MuSiQue dev predictions | Completed on Kaggle |
| [`02_cross_domain_eval.ipynb`](02_cross_domain_eval.ipynb) | Initial small cross-domain evaluation script | Superseded by Notebooks 04 & 05 |
| [`04_five_benchmark_eval.ipynb`](04_five_benchmark_eval.ipynb) | 5-track evaluation run on earlier v5 checkpoint (`6bbd13d5...`) | Completed (`results/04_five_benchmark_v5/`) |
| [`05_full_decision_benchmark_eval.ipynb`](05_full_decision_benchmark_eval.ipynb) | **Full 8-benchmark (9-track) evaluation + post-hoc Temperature Scaling on canonical `93ef195f...`** | **Completed (`results/05_eight_benchmark_eval/`)** |

### Reports & Analysis
- [`reports/eight_benchmark_sota_comparison.md`](reports/eight_benchmark_sota_comparison.md): Detailed 8-benchmark evaluation analysis, v5 vs canonical checkpoint comparison, post-hoc calibration results, and published SOTA comparisons (EMNLP'26, ICLR'25 Sufficient Context, ICLR'25 FaithEval, COLM'24 MultiHop-RAG, ICLR'23 SQuAD2).
- [`reports/bottlenecks_and_literature_gap.md`](reports/bottlenecks_and_literature_gap.md): Plain-language diagnosis of the 5 bottlenecks and mapping to 2024–2026 papers (*Provence, MiniCheck, Sufficient Context, SURE-RAG, UCCI, Verification Without Sufficiency*).
- [`reports/multihop_rag_2025_2026_survey.md`](reports/multihop_rag_2025_2026_survey.md): 10-paper (2025–2026, A*/A + COLM, open-source code) × 10-benchmark comparison matrix & trade-offs.
- [`research_findings.md`](research_findings.md) & [`verification_report.md`](verification_report.md): Research findings summary and upstream audit log.
- [`reproduction-journey.html`](reproduction-journey.html): Interactive visual reproduction timeline.

### Scripts & Configs
- [`scripts/`](scripts/): Training worker (`kaggle_train_worker.py`), notebook generators (`build_full_decision_benchmark_notebook.py`, `build_five_benchmark_notebook.py`, `build_notebooks.py`, `build_15min_eda_notebook.py`), data prep (`prepare_eval_data.py`), Kaggle monitor (`monitor_kaggle_v5.ps1`).
- [`configs/experiment_config.json`](configs/experiment_config.json): Machine-readable experiment configuration and verified metrics.

## Cloning With Checkpoint Weights (Git LFS)

```bash
git lfs install
git clone --branch codex/fact-grounding-reproduction https://github.com/nhthanh1106/3thangngu.git
cd 3thangngu
git lfs pull
```
