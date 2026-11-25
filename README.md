# Tổng hợp đánh giá benchmark: KPR Retrieval & Fact-Grounding Classifier

Repo này tổng hợp mã nguồn tái lập, Kaggle notebooks, checkpoint DeBERTa-v3-large (Git LFS), số liệu máy đọc được (CSV/JSON) và báo cáo phân tích/so sánh SOTA tiếng Việt cho 2 nhóm thực nghiệm retrieval & multi-hop QA.

---

## 1. Reproduction & Đánh giá 8 Benchmark cho Fact-Grounding Classifier (EMNLP 2026)

Toàn bộ notebook huấn luyện, xác minh, đánh giá 8 benchmark (9 track), checkpoint canonical (`SHA-256: 93ef195f...`), kết quả và báo cáo so sánh SOTA nằm trong [`fact_grounding_reproduction/`](fact_grounding_reproduction/).

- **Tài liệu tổng quan & cách tải checkpoint Git LFS:** [`fact_grounding_reproduction/README.md`](fact_grounding_reproduction/README.md)
- **Notebook đánh giá đầy đủ 8 benchmarks:** [`05_full_decision_benchmark_eval.ipynb`](fact_grounding_reproduction/05_full_decision_benchmark_eval.ipynb) (đã chạy `Succeeded` trong 27.3 phút trên Kaggle Tesla T4 × 2)
- **Kết quả máy đọc được:** [`results/05_eight_benchmark_eval/`](fact_grounding_reproduction/results/05_eight_benchmark_eval/) và bản đối chứng v5 [`results/04_five_benchmark_v5/`](fact_grounding_reproduction/results/04_five_benchmark_v5/)
- **Báo cáo phân tích & so sánh SOTA:**
  - [Báo cáo 8-Benchmark & So sánh SOTA (2022–2026)](fact_grounding_reproduction/reports/eight_benchmark_sota_comparison.md)
  - [Giải thích 5 nút thắt & bản đồ paper 2024–2026](fact_grounding_reproduction/reports/bottlenecks_and_literature_gap.md)
  - [Tổng hợp ma trận 10 Paper Multi-Hop RAG (2025–2026, A*/A + code)](fact_grounding_reproduction/reports/multihop_rag_2025_2026_survey.md)

### Tóm tắt kết quả nổi bật (Checkpoint `93ef195f...`, 435M, `14–33 ms/mẫu`)
- **In-domain MuSiQue dev (`6,404` hops):** `F1 = 0.7907` (+0.0057 vs bài gốc `0.785`), `Acc = 0.8142`, `P = 0.8847`, `AUROC = 0.8764`, ECE giảm `0.1176 → 0.0372` sau Temperature Scaling (`T = 1.929`).
- **Tách bạch Format Shift vs Domain Shift:** `MuSiQue-Full` dạng sub-question (`02b`) đạt **AUROC `0.8986` / F1 `0.8375`**, còn dạng full-question (`02a`) giảm về **AUROC `0.7755`**; trong khi chuyển hẳn sang **tin tức MultiHop-RAG (COLM'24)** vẫn đạt **AUROC `0.8911`**, chặn đúng **`96.3%` (`290/301`) `null_query`**.
- **Pairwise Sufficiency Ranking:** `HotpotQA` đạt **`88.55%`**, `FaithEval (ICLR'25)` đạt **`81.83%`**, `2Wiki` đạt **`72.05%`** (trên `10,094` cặp: **`81.02%`**).

---

## 2. KPR và BGE trên BEIR SciFact

Đánh giá retrieval nhẹ trên mẫu kiểm soát của **BEIR SciFact** (test qrels, 40 truy vấn, 800 passage ứng viên, seed 42), so sánh `knowledgeable-ai/kpr-bge-base-en-v1.5` với baseline `BAAI/bge-base-en-v1.5` trên `cuda:0` (float32).

| Metric | BGE | KPR | KPR − BGE |
|---|---:|---:|---:|
| Recall@1 | 0.6958 | 0.6708 | −0.0250 |
| Recall@5 | 0.8750 | 0.8250 | −0.0500 |
| Recall@10 | 0.9000 | 0.9000 | 0.0000 |
| MRR@10 | 0.8069 | 0.7778 | −0.0292 |
| nDCG@10 | 0.8312 | 0.8078 | −0.0233 |

### Các file KPR
- [`kpr-vs-bge-scifact-sampled-evaluation.ipynb`](kpr-vs-bge-scifact-sampled-evaluation.ipynb): notebook Kaggle tự chạy kèm output log đầy đủ.
- [`kpr_eval_report.json`](kpr_eval_report.json): báo cáo cấu trúc đầy đủ (`execution.status = "success"`).
- [`src/`](src/) (`data.py`, `models.py`, `evaluation.py`, `reporting.py`) & [`requirements.txt`](requirements.txt): các module helper tái sử dụng cho pipeline đánh giá KPR.
