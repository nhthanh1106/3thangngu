# Fact-Grounding Classifier Reproduction

This project reproduces the **DeBERTa-v3-large evidence-sufficiency classifier** from [Diagnosing the Fact-Grounding Gap in Multi-Hop Question Answering](https://aclanthology.org/2026.emnlp-main.834/). The verified work covers fine-tuning from the official data, MuSiQue in-domain evaluation, and checkpoint reload verification. It does **not yet reproduce the paper's complete end-to-end QA/re-retrieval comparisons**.

## What the model predicts

For one reasoning hop, the input is a sub-question plus the passages retrieved so far. The classifier predicts whether those passages are sufficient to answer the sub-question:

- `0`: `NOT-ANSWERABLE` / insufficient evidence
- `1`: `ANSWERABLE` / sufficient evidence

This is supervised fine-tuning of a sequence classifier, not fine-tuning a generative model to write answers. The upstream repository's counterfactual labeling code replaces evidence at one hop with gold supporting passages and reruns QA. If that swap fixes the final answer, the original evidence at that hop is labeled insufficient; otherwise it is labeled sufficient. The packaged CSVs only contain `text,label`, so they do not retain the source QA ID or hop ID for row-level tracing.

## Data and training

The notebook uses the official preformatted MuSiQue CSVs and preserves their train/dev split:

| Split | Rows | Unit |
|---|---:|---|
| Train | 46,610 | Hop-level `(sub-question, passages, label)` examples |
| Dev | 6,404 | Hop-level `(sub-question, passages, label)` examples |

These are **not counts of unique full QA questions**. One multi-hop QA question can produce multiple hop examples; the CSVs lack IDs needed to recover the number of parent questions. Exact formatted-text overlap between train and dev is zero. Duplicates within each split are preserved to match the official files.

Verified setup: `microsoft/deberta-v3-large`, maximum length 512, learning rate `2e-5`, effective batch size 16, seed 42, positive-class F1 for best-checkpoint selection, and decision threshold 0.5. Kaggle execution used two Tesla T4 GPUs and Transformers 4.57.1. The paper used A100 hardware; this software/hardware difference is a reproduction deviation.

## Verified MuSiQue dev result

Fresh inference from the saved checkpoint processed all 6,404 dev rows:

| Metric | Paper | Reproduced | Difference |
|---|---:|---:|---:|
| Precision | 0.805 | 0.8847 | +0.0797 |
| Recall | 0.765 | 0.7148 | -0.0502 |
| F1 | 0.785 | 0.7907 | +0.0057 |
| Accuracy | 0.794* | 0.8142 | +0.0202* |

The F1 is close to the paper result, while the precision/recall balance differs. The paper's 0.794 accuracy is reported for a different table/setting, so treat that accuracy comparison cautiously.

The fresh checkpoint reload check passed on three deterministic examples: maximum probability difference `2.24e-8` against a tolerance of `0.001`. The full dev inference took about 700.7 seconds on Kaggle T4 x2.

**Important artifact finding:** the older saved prediction CSV has the correct dev text/labels/order but does not match the selected checkpoint's predictions (maximum probability difference `0.9873`; its F1 is `0.4393`). Use the recomputed outputs from the verification run, not that stale prediction file. The cause of the mismatch has not been established.

## Notebooks

| File | Purpose | Status |
|---|---|---|
| `01_train_fact_grounding_live.ipynb` | Fine-tune the classifier with streamed progress logs and recovery checkpoints | Completed; selected checkpoint is included under `checkpoint/` |
| `03_verify_persisted_checkpoint.ipynb` | Reload the saved checkpoint and recompute full MuSiQue dev predictions/metrics | Completed on Kaggle |
| `02_cross_domain_eval.ipynb` | Evaluate the same checkpoint on prepared datasets without fine-tuning | Prepared; no verified cross-domain run metrics yet |
| `kaggle_15min_eda/15min_eda_pipeline.ipynb` | Short EDA and question-only diagnostic | Prepared; metadata currently points to the older `nguynbquangminh` Kaggle slug and must be aligned before running under `huthnhnguyn` |

## Evaluation data prepared

- MuSiQue: 300 balanced examples for the short EDA, in addition to the verified full-dev run above.
- HotpotQA: 300 paired examples. Each pair uses the same question and source QA, with sufficient versus evidence-deficient passages.
- FiQA-2018: 42 examples, manually judged by one reviewer; this is a small exploratory sample.
- SciFact: not evaluated. Its SUPPORT/CONTRADICT/NOT ENOUGH INFO labels do not directly represent answerability.

No cross-domain metrics are reported until the notebook is run and outputs are saved. HotpotQA also changes the input task from MuSiQue hop-level sub-questions to full questions, so it tests both dataset and task-format shift.

## Kaggle artifacts

Training notebook: [Fact Grounding Training Reproduction (Live Logs)](https://www.kaggle.com/code/huthnhnguyn/fact-grounding-training-reproduction-live-logs)

Verifier run: [Fact Grounding Checkpoint Reload Verification](https://www.kaggle.com/code/huthnhnguyn/fact-grounding-checkpoint-reload-verification/log)

The verifier wrote these files under its Kaggle run's `/kaggle/working/` directory:

- `checkpoint_reload_report.json`
- `recomputed_in_domain_metrics.json`
- `recomputed_in_domain_predictions.csv`
- `checkpoint_reload_samples.csv`

The selected checkpoint files are packaged in [`checkpoint/`](checkpoint/). The main weights file is `model.safetensors` (about 1.74 GB); Git LFS is required to fetch it. The small config and tokenizer files in the same directory are needed to load the classifier.

To clone and retrieve the checkpoint:

```bash
git lfs install
git clone --branch codex/fact-grounding-reproduction https://github.com/nhthanh1106/3thangngu.git
cd 3thangngu
git lfs pull
```

The fine-tuning data and optional cross-domain evaluation data remain linked as Kaggle inputs rather than copied into this GitHub repository. The cross-domain data package has mixed or unspecified source-license metadata; its notebook is included, but its data are not redistributed here.

## Research boundary and next step

The current evidence supports an in-domain classifier reproduction with a successful checkpoint reload. It does not show that adding the decision classifier improves end-to-end QA accuracy or retrieval efficiency, and it does not establish cross-domain generalization. Next, run the paired pipeline comparison on the same QA examples and report final QA EM/F1, retrieval calls, and latency; then run the prepared HotpotQA and FiQA evaluations at the fixed threshold.

## Project files

- `verification_report.md`: paper/repository protocol and data audit.
- `research_findings.md`: research status and limitations.
- `reproduction-journey.html`: visual timeline and result summary.
- `configs/experiment_config.json`: machine-readable configuration.
- `datasets/training_data/`: official train/dev CSVs.
- `kaggle_eval_data/`: prepared cross-domain evaluation package.
- `logs/`: profiling and Kaggle monitoring records.
