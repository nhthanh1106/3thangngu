# Báo Cáo Đánh Giá 8 Benchmark & So Sánh SOTA: Fact-Grounding DeBERTa-v3-Large

- **Phiên chạy Kaggle:** [Fact Grounding 8-Benchmark Decision Eval (Version 1)](https://www.kaggle.com/code/huthnhnguyn/fact-grounding-8-benchmark-decision-eval) · **Trạng thái:** `Succeeded` (`1,641.7s` ≈ **27.3 phút**) · **Phần cứng:** `Tesla T4 × 2` (`Peak VRAM: 5.15 / 4.61 GiB`)
- **Checkpoint xác minh SHA-256:** `93ef195f3840689a70094c8fdbe4c1126a86e9faf1b9382a3040e3d32b470260` (khớp 100% với [`checkpoint/model.safetensors`](../checkpoint/model.safetensors), `1,740,304,440` bytes, `0 = NOT-ANSWERABLE`, `1 = ANSWERABLE`)
- **File kết quả gốc:** [`metrics.csv`](../results/05_eight_benchmark_eval/metrics.csv) · [`metrics_calibrated.csv`](../results/05_eight_benchmark_eval/metrics_calibrated.csv) · [`pairwise_ranking_metrics.csv`](../results/05_eight_benchmark_eval/pairwise_ranking_metrics.csv) · [`run_manifest.json`](../results/05_eight_benchmark_eval/run_manifest.json)

---

## 1. Bảng Kết Quả Đầy Đủ Trên 8 Benchmarks (9 Tracks — Ngưỡng Gốc `0.5`)

| Track | Benchmark | `n_scored` | Acc | Precision | Recall | F1 (Ans) | Macro-F1 | Bal-Acc | **AUROC** | **PR-AUC** | **ECE-15** | Trunc% | Latency (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **01** | **MuSiQue dev (in-domain hops)** | 6,404 | **0.8142** | **0.8847** | 0.7148 | **0.7907** | **0.8118** | **0.8124** | **0.8764** | 0.8913 | **0.1176** | 0.0% | 26.4 |
| **02a** | **MuSiQue-Full (full question)**\* | 2,417 | 0.6967 | 0.7113 | 0.6525 | 0.6806 | 0.6960 | 0.6963 | 0.7755 | 0.7479 | 0.1953 | 50.8% | 29.1 |
| **02b** | **MuSiQue-Full (sub-q hops)**\* | 6,404 | **0.7943** | **0.9531** | 0.7469 | **0.8375** | 0.7788 | **0.8286** | **0.8986** | **0.9570** | 0.1547 | 45.0% | 27.2 |
| **03** | **IIRC dev (incomplete multi-hop)** | 1,301 | **0.7648** | **0.8484** | **0.8270** | **0.8376** | 0.7057 | 0.7104 | 0.7914 | 0.8976 | **0.1403** | 25.7% | 25.7 |
| **04** | **HotpotQA distractor pairs** | 8,018 | 0.6737 | 0.6164 | **0.9199** | 0.7382 | 0.6527 | 0.6737 | **0.8017** | 0.7881 | 0.2651 | 0.0% | 13.8 |
| **05** | **2WikiMultiHopQA pairs** | 7,188 | 0.5716 | 0.5939 | 0.4533 | 0.5141 | 0.5656 | 0.5716 | 0.6012 | 0.5874 | 0.3287 | **95.9%** | 32.8 |
| **06** | **FaithEval Unanswerable pairs** | 4,982 | 0.6345 | 0.5978 | 0.8222 | 0.6922 | 0.6211 | 0.6345 | 0.7215 | 0.7165 | 0.2632 | 50.9% | 27.9 |
| **07** | **MultiHop-RAG (News, COLM'24)** | 2,556 | 0.7148 | **0.9929** | 0.6816 | **0.8083** | 0.6257 | **0.8225** | **0.8911** | **0.9850** | 0.2794 | 11.8% | 26.7 |
| **08** | **SQuAD 2.0 dev (single-hop ctrl)** | 11,873 | 0.6377 | 0.5832 | **0.9620** | 0.7262 | 0.5956 | 0.6382 | **0.8409** | 0.8281 | 0.3150 | 4.3% | 16.6 |

> **Ghi chú minh bạch (`*` ở Track 02a & 02b):** Trong file gốc `musique_full_v1.0_dev.jsonl`, hai bản đối chứng `answerable=True` và `answerable=False` dùng chung cùng một trường `id` (`qid`). Cơ chế dedup theo `example_id` ở Version 1 giữ lại đúng 1 bản ghi cho mỗi `qid` (`2,417/4,834` ở `02a` với tỷ lệ nhãn cân bằng `49.52%` dương, và `6,404/12,808` ở `02b`). Lỗi trùng ID đã được sửa sẵn trong [`scripts/build_full_decision_benchmark_notebook.py`](../scripts/build_full_decision_benchmark_notebook.py) và [`05_full_decision_benchmark_eval.ipynb`](../05_full_decision_benchmark_eval.ipynb) cho các lần chạy sau.

---

## 2. Xếp Hạng Cặp (Pairwise Ranking) & Đối Chiếu Với Checkpoint v5 Cũ (`6bbd13d5`)

| Benchmark | Số cặp | **Checkpoint Canonical `93ef195f` (Notebook 05)** | **Checkpoint v5 `6bbd13d5` (Notebook 04)** | Chênh lệch (`93ef` − `6bbd`) |
|---|---:|---|---|---|
| `01_musique_dev` *(in-domain)* | — | Acc **81.42%** · P **88.47%** · F1 **79.07%** · ECE **0.1176** | Acc 78.78% · P 77.09% · F1 78.90% · ECE 0.1378 | **+2.64% Acc · +11.38% P · −0.020 ECE** |
| `04_hotpotqa_pairs` | 4,009 *(v5: 4,544)* | **Pairwise Acc 88.55%** · AUROC 0.8017 · FP 2,295 | Pairwise Acc 89.56% · AUROC 0.8390 · FP 2,246 | Tương đương *(4,009 cặp sạch hơn)* |
| `05_2wiki_support_pairs` | 3,594 | **Pairwise Acc 72.05%** · AUROC **0.6012** · **FP 1,114** | Pairwise Acc 66.29% · AUROC 0.5862 · FP 2,166 | **+5.76% Pairwise · giảm 48.6% FP** |
| `06_faitheval_pairs` | 2,491 | **Pairwise Acc 81.83%** · Acc **63.45%** · **ECE 0.2632** | Pairwise Acc 79.99% · Acc 61.76% · ECE 0.3064 | **+1.84% Pairwise · +1.69% Acc · −0.043 ECE** |
| **ALL MATCHED PAIRS** | **10,094** | **Pairwise Acc 81.02%** · Mean logit margin `+1.966` | Pairwise Acc 79.45% | **+1.57% Pairwise** |

---

## 3. Tác Động Của Post-Hoc Temperature Scaling (Fit Duy Nhất Trên `01_musique_dev` — 0 Rò Rỉ OOD)

Fit tham số nhiệt độ duy nhất `T = 1.9290` trên `01_musique_dev` (giảm NLL từ `0.5272` → `0.4318`) rồi đóng băng áp sang toàn bộ 8 track còn lại:

| Track | ECE gốc (`T=1`) | **ECE sau Temp Scaling (`T=1.929`)** | Mức giảm ECE | F1 tại `thr=0.5` | F1 tại `thr=0.36` *(fit trên MuSiQue)* |
|---|---:|---:|---:|---:|---:|
| `01_musique_dev` | 0.1176 | **0.0372** | **−68.4%** | 0.7907 | **0.7973** |
| `02a_musique_full_q` | 0.1953 | **0.0853** | **−56.3%** | 0.6806 | **0.7139** (+3.33%) |
| `02b_musique_full_hops` | 0.1547 | **0.1482** | −4.2% | 0.8375 | **0.8610** (+2.35%) |
| `03_iirc_dev` | 0.1403 | **0.0776** | **−44.7%** | 0.8376 | **0.8593** (+2.17%) |
| `04_hotpotqa_pairs` | 0.2651 | **0.1957** | **−26.2%** | **0.7382** | 0.7262 *(tăng FP)* |
| `05_2wiki_support_pairs` | 0.3287 | **0.2165** | **−34.1%** | 0.5141 | **0.5517** (+3.76%) |
| `06_faitheval_pairs` | 0.2632 | **0.1688** | **−35.9%** | **0.6922** | 0.6897 *(tăng FP)* |
| `07_multihop_rag_news` | 0.2794 | 0.3055 | +9.3% | 0.8083 | **0.9077** (+9.94%, Acc 84.3%) |
| `08_squad_v2_dev` | 0.3150 | **0.2528** | **−19.7%** | **0.7262** | 0.7098 *(tăng FP)* |

---

## 4. So Sánh Tổng Thể Với Các Phương Pháp SOTA Đã Công Bố (2022–2026)

| Benchmark | Kết quả Checkpoint DeBERTa (435M, Zero-shot OOD, `14–33 ms`) | Mốc SOTA / Baseline Đã Công Bố Trong Literature | Đánh Giá Vị Thế & Trade-off |
|---|---|---|---|
| **MuSiQue dev (hop-level)** | **F1 79.07% · Acc 81.42% · P 88.47% · AUROC 0.876** *(ECE 0.037 sau T-scale)* | **Fact-Grounding Gap (EMNLP 2026 — bài gốc):** DeBERTa-v3-large `F1 78.5%`, `P 80.5%`, `R 76.5%`, `Acc 79.4%*` | **Tái lập vượt nhẹ bài gốc** (+0.57% F1, +7.97% Precision, +2.02% Acc). |
| **Multi-Hop Context Sufficiency** *(MuSiQue-Full hops `02b` vs full-q `02a`)* | **Sub-q hops (`02b`):** `Acc 79.4%` (`81.8%` cal), **`F1 83.7%`** (`86.1%` cal), **`AUROC 0.899`**<br>**Full-q (`02a`):** `Acc 69.7%`, `F1 68.1%`, `AUROC 0.776` | **Sufficient Context (ICLR 2025 — Google):** `Gemini 1.5 Pro (1-shot CoT)` = **Acc 93.0% / F1 94.0%**; `FLAMe-24B` = **Acc 87.8%**; `TRUE-NLI` = **Acc 82.6%**; `Contains-GT` = **Acc 80.9%**.<br>**MuSiQue-Full (TACL 2022):** `EX(SA)` joint `An+Sf 32.2` / `Sp+Sf 44.3`. | Ở đúng format **sub-question (`02b`)**, encoder 435M tiệm cận `TRUE-NLI` / `FLAMe-24B` nhưng **nhỏ hơn 55×–1000×, chạy ~27 ms/mẫu, chi phí API = $0**. Sang full-question (`02a`), khoảng cách tăng mạnh → cần tách sub-question trước khi qua DeBERTa. |
| **IIRC (Incomplete Multi-Hop)** | **Acc 76.5%** (`78.4%` cal) · **F1 83.8%** (`85.9%` cal) · **AUROC 0.791** · **PR-AUC 0.898** · ECE `0.078` *(sau T-scale)* | **IIRC (EMNLP 2020):** `NumNet+ (RoBERTa-base)` pipeline đạt `31.1%` overall QA F1 (nhận diện `none`/unanswerable là nút cổ chai chính). | **Chuyển giao rất mạnh** sang tập incomplete-info chuẩn mực mà không cần fine-tune lại. |
| **HotpotQA pairs** | **Pairwise Acc 88.55%** · **AUROC 0.802** · F1 73.8% · Acc 67.4% | Reranker / LLM judge mạnh trên HotpotQA thường duyệt toàn bộ 2–10 đoạn văn dài. | **Xếp hạng cặp cực mạnh (`88.6%`)** với độ trễ `13.8 ms/mẫu`, nhưng ngưỡng cố định `0.5` thiên dương nặng (`Recall 92.0%`, `FP 2,295`). |
| **2WikiMultiHopQA pairs** | **Pairwise Acc 72.05%** · AUROC 0.601 · Acc 57.2% · F1 51.4% | Full-context LLM / Graph-RAG không bị giới hạn 1,500 ký tự. | **Bị giới hạn input:** `95.9%` mẫu bị cắt cụt ở `1,500` ký tự (`char_trunc_rate = 0.959`), xóa mất câu bằng chứng nằm ở phần sau context. |
| **FaithEval Unanswerable (ICLR 2025)** | **Pairwise Acc 81.83%** · AUROC 0.722 · F1 69.2% · Acc 63.5% (`Recall 82.2%`, `FP 1,378`) | **FaithEval (ICLR 2025 — Salesforce):** Frontier LLM + CoT đạt đỉnh **~71.8% accuracy** trên unanswerable context và tụt `13.6%–68.4%` so với context gốc. | Dù chỉ 435M và bị cắt cụt `50.9%` mẫu ở 1,500 chars, DeBERTa vẫn phân biệt cặp đúng **`81.83%`** và chỉ kém đỉnh CoT frontier LLM ~8 điểm accuracy. |
| **MultiHop-RAG News (COLM 2024)** | **AUROC 0.891** · **PR-AUC 0.985** · **Precision 99.3%** · **Bal-Acc 82.3%** · **Bắt đúng `null_query`: `96.3%` (`290/301`)** *(F1 `90.8%` / Acc `84.3%` tại `thr=0.36`)* | **MultiHop-RAG (COLM 2024):** `GPT-4` đạt `56%` acc (retrieved) / `89%` (gold); `Llama-2-70B` đạt `28%–32%`. | **Điểm sáng lớn nhất:** chuyển hoàn toàn sang **tin tức thực tế** ngoài Wikipedia nhưng đạt **`AUROC 0.891`** và chặn tới **`96.3%` `null_query`** (chỉ 11 FP). |
| **SQuAD 2.0 dev (Single-Hop Control)** | **AUROC 0.841** · Recall 96.2% · F1 72.6% · **Acc 63.8%** (`FP = 4,076`) | **DeBERTa-v3-large SQuAD2 SOTA (ICLR 2023):** `NoAns_f1 = 90.83%–93.68%`, overall F1 `~91.5%`. | Khoảng cách ~27 điểm Acc (dù **AUROC vẫn cao `0.841`**) phơi bày điểm yếu chí mạng trước **bẫy adversarial lexical overlap**. |

Xem thêm giải thích chi tiết 5 nút thắt và ánh xạ sang các paper 2024–2026 tại [`bottlenecks_and_literature_gap.md`](bottlenecks_and_literature_gap.md).
