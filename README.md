# Tổng hợp đánh giá benchmark

Repo này lưu notebook, checkpoint, báo cáo kết quả máy đọc được (CSV/JSON) và ghi chú tiếng Việt cho các phép đánh giá retrieval & multi-hop QA. Mỗi benchmark có notebook chạy được, file kết quả và README giải thích dữ liệu, cách chấm cùng giới hạn của kết luận.

---

## 1. Reproduction & Đánh giá 8 Benchmark cho Fact-Grounding Classifier (EMNLP 2026)

Notebook, checkpoint DeBERTa-v3-large (`SHA-256: 93ef195f...` qua Git LFS), kết quả đánh giá 8 benchmark (9 tracks) và báo cáo so sánh SOTA được lưu trong [`fact_grounding_reproduction/`](fact_grounding_reproduction/).

- **README hướng dẫn & cách lấy checkpoint Git LFS:** [`fact_grounding_reproduction/README.md`](fact_grounding_reproduction/README.md)
- **Notebook đánh giá đầy đủ 8 benchmarks:** [`05_full_decision_benchmark_eval.ipynb`](fact_grounding_reproduction/05_full_decision_benchmark_eval.ipynb) (chạy `Succeeded` trong 27.3 phút trên Kaggle Tesla T4 × 2)
- **Kết quả máy đọc được:** [`results/05_eight_benchmark_eval/`](fact_grounding_reproduction/results/05_eight_benchmark_eval/) (bản canonical `93ef195f...`) và bản đối chứng v5 [`results/04_five_benchmark_v5/`](fact_grounding_reproduction/results/04_five_benchmark_v5/)
- **Báo cáo phân tích & so sánh SOTA:**
  - [Báo cáo 8-Benchmark & So sánh SOTA (2022–2026)](fact_grounding_reproduction/reports/eight_benchmark_sota_comparison.md)
  - [Giải thích 5 nút thắt & bản đồ paper 2024–2026](fact_grounding_reproduction/reports/bottlenecks_and_literature_gap.md)
  - [Tổng hợp ma trận 10 Paper Multi-Hop RAG (2025–2026, A*/A + code)](fact_grounding_reproduction/reports/multihop_rag_2025_2026_survey.md)

### Tóm tắt kết quả nổi bật (Checkpoint canonical `93ef195f...`, 435M, `14–33 ms/mẫu`)
- **In-domain MuSiQue dev (`6,404` hops):** `F1 = 0.7907` (+0.0057 vs bài gốc `0.785`), `Acc = 0.8142`, `P = 0.8847`, `AUROC = 0.8764`, ECE giảm `0.1176 → 0.0372` sau Temperature Scaling (`T = 1.929`).
- **Tách bạch Format Shift vs Domain Shift:** `MuSiQue-Full` dạng sub-question (`02b`) đạt **AUROC `0.8986` / F1 `0.8375`**, còn dạng full-question (`02a`) giảm về **AUROC `0.7755`**; trong khi chuyển hẳn sang **tin tức MultiHop-RAG (COLM'24)** vẫn đạt **AUROC `0.8911`**, chặn đúng **`96.3%` (`290/301`) `null_query`**.
- **Pairwise Sufficiency Ranking:** `HotpotQA` đạt **`88.55%`**, `FaithEval (ICLR'25)` đạt **`81.83%`**, `2Wiki` đạt **`72.05%`** (trên `10,094` cặp: **`81.02%`**).

### Lượt tái lập độc lập trên tài khoản `hungluongduc`
Một lượt Kaggle **riêng** trên tài khoản `hungluongduc` đã chạy đủ ba epoch và đạt MuSiQue dev `F1 = 0.7936` ở checkpoint epoch 2, so với `0.785` trong paper. Xem [báo cáo kết quả và các khác biệt về protocol](reports/fact-grounding-gap-deberta-musique-kaggle-results.md). Checkpoint của lượt này nằm ở output Kaggle stage 1, không phải checkpoint trong `fact_grounding_reproduction/`.

---

## 2. KPR và BGE trên BEIR SciFact

Đây là phép đánh giá retrieval nhẹ trên một mẫu của **BEIR SciFact**, dùng test split qrels. Nó so sánh checkpoint `knowledgeable-ai/kpr-bge-base-en-v1.5` với baseline `BAAI/bge-base-en-v1.5`.

**Phạm vi:** Notebook này chỉ chạy phép so sánh inference giữa hai checkpoint có sẵn. KPR thuộc paper *Dynamic Injection of Entity Knowledge into Dense Retrievers*, khác sáu paper nhóm trình bày (TRA-RAG, HEAR-RAG, SG-RAG, FACTA, CIRAG, Fact-Grounding Gap). Notebook KPR không benchmark Fact-Grounding Gap với SRO–TRA–uncertainty. Xem [báo cáo chi tiết về model, kết quả và mức độ tái tạo](reports/kpr-bge-scifact-model-benchmark.md).

### Dữ liệu và cách lấy mẫu

- Có 300 truy vấn SciFact đủ nhãn; notebook chọn 40 truy vấn bằng seed 42.
- Mỗi truy vấn được xếp hạng trên cùng một tập 800 passage ứng viên (49 cặp truy vấn–passage liên quan, 44 passage liên quan duy nhất và 756 passage gây nhiễu).
- Đây là **restricted candidate pool**, không phải chạy toàn bộ corpus SciFact như một benchmark BEIR đầy đủ.
- Qrels đánh giá mức độ liên quan của bằng chứng với claim; không gắn nhãn passage có ủng hộ hay phản bác claim hay không.

### Kết quả (`cuda:0`, float32, 768 chiều)

| Metric | BGE | KPR | KPR − BGE |
|---|---:|---:|---:|
| Recall@1 | 0.6958 | 0.6708 | −0.0250 |
| Recall@5 | 0.8750 | 0.8250 | −0.0500 |
| Recall@10 | 0.9000 | 0.9000 | 0.0000 |
| MRR@10 | 0.8069 | 0.7778 | −0.0292 |
| nDCG@10 | 0.8312 | 0.8078 | −0.0233 |

Trong mẫu này, KPR bằng BGE ở Recall@10 nhưng thấp hơn ở các metric còn lại. Kết quả chưa đủ để kết luận model nào tốt hơn nói chung: chỉ có 40 truy vấn, tập ứng viên bị giới hạn và chưa có khoảng tin cậy hay kiểm định ý nghĩa thống kê.

### Các file KPR
- [`kpr-vs-bge-scifact-sampled-evaluation.ipynb`](kpr-vs-bge-scifact-sampled-evaluation.ipynb): notebook Kaggle tự chạy kèm log và kết quả.
- [`kpr_eval_report.json`](kpr_eval_report.json): báo cáo cấu trúc đầy đủ (`execution.status = "success"`).
- [`src/`](src/) (`data.py`, `models.py`, `evaluation.py`, `reporting.py`) & [`requirements.txt`](requirements.txt): các module helper tái sử dụng cho pipeline đánh giá KPR.
- [`reports/kpr-bge-scifact-model-benchmark.md`](reports/kpr-bge-scifact-model-benchmark.md): báo cáo phân tích chi tiết KPR vs BGE.
