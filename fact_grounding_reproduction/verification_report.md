# Verification report

## Sources checked

- Paper: [Diagnosing the Fact-Grounding Gap in Multi-Hop Question Answering](https://aclanthology.org/2026.emnlp-main.834/), including its PDF and Appendix I.
- Official source repository: [RichardZhu123/fact-grounding-gap](https://github.com/RichardZhu123/fact-grounding-gap), checked at commit `3df77c2c94863ff35b398e14b37bd1e130e9d6de`.
- Inspected `train_deberta.py`, `deberta_inference.py`, `deberta_fg_train.csv`, and `deberta_fg_dev.csv`. Also inspected `format_deberta_data.py` to verify input construction.
- GitHub Releases API returned an empty release list. Hugging Face model search for `fact-grounding-gap` returned no models. No official fine-tuned checkpoint was found, so this reproduction trains from the pretrained base model.

## Verified model and protocol

| Item | Verified value |
|---|---|
| Base model | `microsoft/deberta-v3-large` |
| Architecture | DeBERTa-v3-large, 24 layers, hidden size 1,024, approximately 435M parameters |
| Task | Binary classification of `(sub-question, accumulated passages)` as NOT-ANSWERABLE (0) or ANSWERABLE (1) |
| Serialization | `Sub-question: {question}\n\nPassages:\n{passage text}`; CSVs already contain formatted text |
| Passage truncation | 1,500 characters in the data formatter; full serialized string is longer because of question and prefix |
| Tokenization | DeBERTa tokenizer; max sequence length 512; dynamic batch padding to a multiple of 8; truncation |
| Training rows | 46,610 MuSiQue hop-level examples |
| Development rows | 6,404 held-out MuSiQue hop-level examples |
| Split | Official train/dev split retained; zero exact formatted-text overlap |
| Learning rate | `2e-5` |
| Effective batch size | 16; implementation adjusts micro-batch and gradient accumulation to detected Kaggle GPUs |
| Gradient checkpointing | Version 7 tests enabled checkpointing with `use_reentrant=False`; version 4's default reentrant mode failed during backward propagation. This is a documented hardware adaptation, not specified in the paper's reference configuration. |
| Epochs | Paper says convergence at epoch 3; official script sets maximum 10 epochs |
| Optimizer | Hugging Face Trainer default AdamW implementation; official script does not override optimizer |
| Scheduler | Linear with 10% warmup |
| Weight decay | 0.01 (official training script) |
| Selection | Dev F1, `greater_is_better=True`, early stopping patience 3 |
| Threshold | 0.5 for ANSWERABLE |
| Metrics | Positive-class precision, recall, F1, accuracy |
| Seed | 42 |

The Hugging Face model card lists the base model under the MIT license ([microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large)). Kaggle's dataset metadata schema does not list MIT as a selectable license value; if the fine-tuned weights are persisted there, use its `other` value and state the upstream MIT license in the dataset description ([Kaggle CLI metadata documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md)). The checkpoint dataset will remain private.

## Data integrity findings

The repository CSVs contain exactly the paper's stated row counts and only `text,label` columns; they have no IDs, so duplicate-ID checks are not possible. The exact text overlap between train and dev is zero. The training split contains duplicate text rows and some identical texts with conflicting labels; the same is true in dev. These rows are preserved because silently deduplicating or relabeling would diverge from the official data. Counts are recorded by the notebook at runtime.

The prepared evaluation examples have unique IDs. MuSiQue's synthetic IDs hash the original dev row index together with its text, so duplicate source text rows remain distinct; HotpotQA IDs retain source question IDs with positive/negative suffixes, and FiQA IDs combine query IDs with pair labels.

## Paper reference metrics

The paper's Table 4 reports DeBERTa-v3-large dev precision `0.805`, recall `0.765`, and F1 `0.785` on all 6,404 MuSiQue dev hops. Table 5 reports accuracy `79.4%` for LLM-judge labels with passages (and F1 `0.785`). These values are the comparison targets. Reproduced values remain pending until the Kaggle training run completes; this report does not claim a completed reproduction.

## Differences to disclose

- The repository asks for Transformers 4.40.0, but Kaggle's current Python 3.13 runtime cannot build that release's `tokenizers` dependency. The first Kaggle attempt failed during that build, before importing the training code. The notebook now pins Transformers 4.57.1, whose Trainer API is supported by the runtime, and explicitly sets `optim='adamw_torch'` to preserve the official script's optimizer choice. This software-version deviation will be recorded with the actual Kaggle environment.
- The paper reports A100 hardware. The Kaggle run will record the actual accelerator(s), memory, GPU utilization visible to PyTorch, measured 100-step time, and total training time.
- The executed Kaggle runtime reported two Tesla T4 GPUs, 14.56 GiB each. Training v3 exhausted memory at per-device micro-batches 8 and 4 and stopped before full training. Training v4 reached backward propagation but failed inside PyTorch's reentrant gradient checkpointing with “Trying to backward through the graph a second time”; it also stopped before full training.
- Training v5 disabled gradient checkpointing and used fixed max-length padding. Its 100-step profile succeeded at micro-batch 2 per GPU with gradient accumulation 4, taking 559.30 seconds (5.593 seconds/optimizer step; peak allocated memory 11.735 and 6.868 GiB on the two GPUs). The estimated time through epoch 3 with 25% overhead was 16.98 hours. The notebook stopped before full fine-tuning under its preflight guard, so there is no v5 checkpoint or metric.
- Training v6 retained the complete data, effective batch 16, and 512-token truncation limit, and switched to dynamic padding (batch longest, padded to a multiple of 8). The measured 100-step profile took 440.99 seconds (4.410 seconds/optimizer step) at micro-batch 2 per GPU with gradient accumulation 4; peak allocated memory was 11.161 and 9.247 GiB. Mean token length was about 331, with train p99 417 and max 505. Despite reduced padding, projected time through epoch 3 with 25% overhead remained 13.39 hours. The preflight guard stopped before full fine-tuning; no v6 checkpoint or metrics exist.
- Training v7 kept all data, effective batch 16, and the 512-token cap, and tested dynamic padding with non-reentrant gradient checkpointing. The 100-step profile passed at per-device batch 8, accumulation 1, and effective batch 16. It took 401.93 seconds (4.019 seconds/step), with reported peak memory 8.148 and 2.689 GiB across the two GPUs. It projected 12.20 hours through epoch 3 with 25% overhead and stopped before fine-tuning. PyTorch recommends the non-reentrant implementation, which supports backward operations that the reentrant variant restricts ([PyTorch checkpoint documentation](https://docs.pytorch.org/docs/stable/checkpoint.html)); Transformers 4.57.1 exposes `gradient_checkpointing_kwargs` in `TrainingArguments` ([Transformers 4.57.1 Trainer docs](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer)). The reported GPU memory peak is asymmetric, so version 8 logs Trainer's actual parallel mode and batch size and resets memory statistics separately for each device.
- Training v8 confirmed `torch.nn.DataParallel` over device IDs 0 and 1, `n_gpu=2`, train batch 16, and effective batch 16. Length-grouped dynamic padding with non-reentrant gradient checkpointing profiled at batch 8/GPU and accumulation 1, taking 375.19 seconds (3.752 seconds/step); reported peak memory was 8.151 and 2.753 GiB. The projected time through epoch 3 with 25% overhead was 11.39 hours, so preflight stopped before full fine-tuning. The memory figures remain asymmetric despite both device IDs appearing in the wrapper.
- Training v9 under `nguynbquangminh` launches explicit two-process DistributedDataParallel via `torchrun` rather than single-process DataParallel. It keeps full data, effective batch 16, 512-token cap, dynamic padding, length grouping, and non-reentrant checkpointing. At 09:21:32 Asia/Bangkok, Kaggle still reported `RUNNING` after 4 hours 20 minutes, but the CLI returned no notebook logs or output files. The old notebook uses captured subprocess output, which explains silence but cannot distinguish normal work from a hang. No v9 profile, full-training metrics, or checkpoint have been verified.
- A separate live-log notebook was imported and saved as Version 1 under the Kaggle browser account [huthnhnguyn](https://www.kaggle.com/code/huthnhnguyn/fact-grounding-training-reproduction-live-logs), then started with GPU T4 x2 and Internet enabled. At 09:43 Asia/Bangkok, the visible trainer log was loss 0.5422, grad norm 4.6265, epoch 0.09952. This suggests progress beyond the configured 100-step profile, but the explicit `full_training_begin` marker was not visible, so the transition to full fine-tuning is unconfirmed. The notebook streams subprocess output, emits a 30-second heartbeat, logs training every 10 optimizer steps, and saves recoverable model checkpoints per epoch. The session display name is Hữu Thành Nguyễn, but its URL slug `huthnhnguyn` differs from the requested spelling `huuthanhngyn`.
- Numerical differences can result from precision, GPU kernels, library versions, and training nondeterminism.

## Cross-domain protocol constraint

SciFact labels are claim-verification SUPPORT/CONTRADICT/NOT ENOUGH INFO labels. They are not converted directly into answerability labels; SciFact remains not evaluated because no independently reviewed QA conversion was prepared.

The prepared evaluation set contains 300 stratified MuSiQue dev rows (150 per class), 300 HotpotQA examples (150 paired source questions), and 42 FiQA examples (21 manually reviewed query/passage pairs). HotpotQA positives include both annotated supporting sentences. Each negative removes one support sentence and substitutes a length-matched non-support sentence from that source example. A deterministic 20-pair sample was manually inspected after construction, and ambiguous or answer-leaking cases found during review were excluded in the generator. This is a limited quality check, not an inter-annotator study.

FiQA test qrels are used only to select an on-topic positive candidate; they are not mapped to the classifier labels. A single reviewer judged each passage for whether it directly supplies the requested answer and recorded a rationale. There is no second rater. MTEB lists FiQA's license as not specified, so the derived evaluation package remains private and its Kaggle license metadata is `unknown`. The 42 examples are intentionally below the suggested 200 because only a smaller subset received manual answerability review.

Requested sample sizes were 300 MuSiQue, 300 HotpotQA, 200 SciFact, and 200 FiQA. Achieved prepared counts are 300, 300, 0, and 42 respectively. No SciFact metric should be reported. The fixed model threshold remains 0.5. This evaluation still differs from MuSiQue: HotpotQA questions are full questions rather than decomposed hop sub-questions, and the evidence sets are constructed from sentence-level support annotations.
