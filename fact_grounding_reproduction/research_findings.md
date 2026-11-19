# Research findings

## Status

At 09:21:32 Asia/Bangkok on 2026-11-18, the earlier CLI-account training version 9 was still `RUNNING` after 4 hours 20 minutes. The Kaggle CLI returned no notebook logs or output files, so its stage remains unconfirmed. A second notebook was uploaded to the account active in the browser side tab and Version 1 began running on GPU T4 x2. By 09:43 Asia/Bangkok, its visible trainer log was loss 0.5422, grad norm 4.6265, epoch 0.09952. This epoch fraction suggests progress beyond the configured 100-step profile, but no explicit `full_training_begin` marker was visible, so the fine-tuning transition remains unconfirmed. The notebook streams subprocess output, emits a 30-second heartbeat, logs training every 10 optimizer steps, and saves recovery checkpoints by epoch. No completed training metrics or checkpoint are verified yet. The browser profile displays Hữu Thành Nguyễn; its URL slug is `huthnhnguyn`, not the requested spelling `huuthanhngyn`.

This is a live report. Kaggle training versions 3 through 8 all stopped before full fine-tuning and produced no checkpoint or reproduction metrics. The run detected two Tesla T4 GPUs with 14.56 GiB each. Version 3 exhausted memory at per-device micro-batches 8 and 4. Version 4 failed in PyTorch reentrant gradient checkpointing during backward propagation. Version 5's fixed-padding profile took 5.593 seconds per optimizer step and projected 16.98 hours through epoch 3 with 25% overhead. Version 6's dynamic-padding profile used micro-batch 2, gradient accumulation 4, and effective batch 16. It took 440.99 seconds for 100 optimizer steps (4.410 seconds/step), with peak allocated memory of 11.161 and 9.247 GiB; projected time through epoch 3 with 25% overhead was 13.39 hours. Version 7 used non-reentrant gradient checkpointing, dynamic padding, micro-batch 8, accumulation 1, and effective batch 16. It took 401.93 seconds for 100 optimizer steps (4.019 seconds/step), with reported peak allocated memory of 8.148 and 2.689 GiB; projected time through epoch 3 with 25% overhead was 12.20 hours. Version 8 confirmed DataParallel across both T4s, global batch 16, and length grouping. It took 375.19 seconds for 100 steps (3.752 seconds/step), with reported peak memory of 8.151 and 2.753 GiB; projection through epoch 3 with overhead was 11.39 hours. The two-GPU memory values remain asymmetric. The safety guard stopped versions 5–8 before full training; no checkpoint was saved. Version 9 was pushed at 05:00 Asia/Bangkok to launch two-process DDP explicitly while preserving all training data, the 512-token cap, and effective batch 16. The profile or later training stage is not visible via the CLI, so the active stage is unconfirmed. The preflight cutoff remains seven projected hours through epoch 3 including 25% overhead, below Kaggle's observed eight-hour notebook limit. If early stopping requires later epochs than the paper's reported convergence at epoch 3, the job could still exceed the platform limit.

## A. Reproduction quality

The paper reports MuSiQue development precision 0.805, recall 0.765, and F1 0.785; the paper also reports 0.794 accuracy for its full-input setting. These are reference values, not reproduction results.

| Metric | Paper | Reproduction | Difference |
|---|---:|---:|---:|
| Precision | 0.805 | Pending | Pending |
| Recall | 0.765 | Pending | Pending |
| F1 | 0.785 | Pending | Pending |
| Accuracy | 0.794* | Pending | Pending |

\*The paper's 0.794 accuracy comes from its full-input table/setting; verify metric comparability before interpreting the difference.

No successful reproduction is claimed until the fine-tuned checkpoint reload check passes and the full MuSiQue dev metrics are available. The browser-side run is still active; its transition past profile is not yet verified, and checkpoint/full dev metrics are pending.

## B. Cross-domain generalization

Notebook 02 has a fixed threshold of 0.5 and uses the persisted checkpoint without fine-tuning or target-domain threshold tuning. The prepared evaluation set contains 300 MuSiQue rows (150 per class), 300 paired HotpotQA examples (150 per class), and 42 FiQA examples (21 per class). SciFact is deliberately not evaluated because claim-verification verdicts do not directly encode whether passages answer a question. Cross-domain metrics are pending the training/checkpoint milestone and subsequent Kaggle execution.

HotpotQA is a task shift as well as a dataset shift: it uses full questions rather than MuSiQue hop sub-questions. Positives contain both annotated support sentences; negatives replace one support sentence with a length-matched non-support sentence. The 20-pair manual spot check and exclusions are documented in `datasets/manual_inspection.md`. FiQA uses a 21-pair single-reviewer answerability subset; its retrieval qrels are not used as binary labels. MTEB marks its license as not specified, so the derived package remains private with license metadata `unknown`.

## C. Failure analysis

No model failure cases are available yet. The evaluation notebook will export per-example predictions and errors. After execution, classify observed errors by evidence omission, entity mismatch, relation mismatch, time/number, negation, domain vocabulary, and confidence calibration where the records support those categories. Do not infer a causal shortcut from aggregate metrics alone.

## D. Research implications

The six requested research questions remain open until the evaluation run completes. Current protocol caveats:

1. Cross-domain performance estimates are limited by the constructed HotpotQA subset and the small, single-reviewed FiQA sample.
2. The question-only ablation is explicitly out-of-distribution and can reveal reliance patterns, but cannot establish that the full model causally ignores evidence.
3. Lower cross-domain performance, if observed, will not by itself prove overfitting or constitute a novel research result.
4. SciFact is not a binary answerability benchmark under its released labels, so no score will be reported for it without a separately validated conversion.

The next research direction should depend on actual errors: if the classifier misses altered relations or dates, test a manually validated counterfactual set; if calibration is weak, evaluate held-out calibration methods without tuning on target test data; if HotpotQA performance drops, distinguish question-format/task shift from domain shift using a compatible multi-hop dataset.
