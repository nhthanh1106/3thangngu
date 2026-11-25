# Tổng Hợp 10 Paper Multi-Hop QA / RAG Impact Cao (2025–2026, Có Code Chính Thức)

Tài liệu tham khảo tổng hợp **10 công trình đã được chấp nhận chính thức giai đoạn 2025–2026** (Rank A*/A & COLM 2025, có mã nguồn công khai trên GitHub), đối chiếu điểm gốc + metric + trade-off cùng dòng tham chiếu **Fact-Grounding Gap (EMNLP 2026)** đang tái lập trong repo này.

## 1. Danh sách mã nguồn chính thức

1. **HippoRAG 2** (*ICML 2025*, A*) — [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)
2. **SubgraphRAG** (*ICLR 2025*, A*) — [Graph-COM/SubgraphRAG](https://github.com/Graph-COM/SubgraphRAG)
3. **HopRAG** (*ACL 2025 Findings*, A*) — [LIU-Hao-2002/HopRAG](https://github.com/LIU-Hao-2002/HopRAG)
4. **KET-RAG** (*KDD 2025*, A*) — [waetr/KET-RAG](https://github.com/waetr/KET-RAG)
5. **CoRAG** (*NeurIPS 2025*, A*) — [microsoft/LMOps (corag)](https://github.com/microsoft/LMOps/tree/main/corag)
6. **Search-o1** (*EMNLP 2025 Oral*, A*) — [RUC-NLPIR/Search-o1](https://github.com/RUC-NLPIR/Search-o1)
7. **Search-R1** (*COLM 2025*, Top LLM) — [PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1)
8. **DeepResearcher** (*EMNLP 2025*, A*) — [GAIR-NLP/DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher)
9. **DeepRAG** (*ICLR 2026*, A*) — [icip-cas/DeepRAG](https://github.com/icip-cas/DeepRAG)
10. **LinearRAG** (*ICLR 2026*, A*) — [DEEP-PolyU/LinearRAG](https://github.com/DEEP-PolyU/LinearRAG)
- *Ref:* **Fact-Grounding Gap** (*EMNLP 2026*, A*) — [RichardZhu123/fact-grounding-gap](https://github.com/RichardZhu123/fact-grounding-gap)

---

## 2. Ma trận 10 Papers (2025–2026) × 10 Benchmarks (Kèm Điểm & Metric Gốc)

> `R@5` = Passage Recall@5 · `Cov` = Context Coverage · `EM` = Exact Match · `F1` = token F1 · `MBE` = Model-Based Eval (GPT-4o-mini judge) · `Hit@1` = KGQA Top-1 · `Avg-Acc` = trung bình (`Contain-Acc`, `GPT-Acc`). Không so ngang điểm giữa các backbone/metric khác nhau.

| Paper (Venue, Backbone) | HotpotQA | 2Wiki | MuSiQue | Bamboogle | PopQA | NQ | TriviaQA | NarrativeQA | CWQ *(KG)* | FRAMES |
|---|---|---|---|---|---|---|---|---|---|---|
| **1. HippoRAG 2** *(ICML'25, Llama-3.3-70B)* | R@5: 96.3 / F1: 75.5 | R@5: 90.4 / F1: 71.0 | R@5: 74.7 / F1: 48.6 | — | F1: 56.2 | F1: 63.3 | — | F1: 25.9 *(LV-Eval: 12.9)* | — | — |
| **2. SubgraphRAG** *(ICLR'25, GPT-4o-mini / 8B)* | — | — | — | — | — | — | — | — | Hit@1: 62.0 / F1: 54.1 *(8B: 57.0/47.1)* | — *(WebQSP Hit@1: 90.1)* |
| **3. HopRAG** *(ACL'25 Find., GPT-3.5)* | EM: 61.3 / F1: 78.3 | EM: 61.6 / F1: 68.9 | EM: 39.1 / F1: 53.0 | — | — | — | — | — | — | — |
| **4. KET-RAG** *(KDD'25, mode P / U)* | Cov: 82.6% *(U: 81.6%)* | — | Cov: 80.2% *(U: 77.0%)* | — | — | — | — | — | — | — |
| **5. CoRAG** *(NeurIPS'25, Llama-3.1-8B BoN-8)* | EM: 56.3 / F1: 69.8 | EM: 72.5 / F1: 77.3 | EM: 30.9 / F1: 42.4 | EM: 54.4 / F1: 68.3 | — | — *(KILT)* | — *(KILT)* | — | — | — |
| **6. Search-o1** *(EMNLP'25 Oral, QwQ-32B)* | EM: 56.7 / F1: 57.7 | ✓ *(EM/F1)* | ~33.0 *(F1/EM)* | ✓ *(EM/F1)* | — | ✓ *(EM/F1)* | ✓ *(EM/F1)* | — | — | — *(GPQA/AIME)* |
| **7. Search-R1** *(COLM'25, Qwen2.5-7B-Base)* | EM: 45.24 | EM: 45.46 | EM: 23.05 | EM: 55.65 | EM: 47.69 | EM: 49.22 | EM: 66.86 | — | — | — |
| **8. DeepResearcher** *(EMNLP'25, Qwen2.5-7B, Web)* | F1: 52.8 / MBE: 64.3 | F1: 59.7 / MBE: 66.6 | F1: 27.1 / MBE: 29.3 *(OOD)* | F1: 71.0 / MBE: 72.8 *(OOD)* | F1: 48.5 / MBE: 52.7 *(OOD)* | F1: 39.6 / MBE: 61.9 | F1: 78.4 / MBE: 85.0 | — | — | ✓ |
| **9. DeepRAG** *(ICLR'26, DeepRAG-RLon)* | EM: 39.10 / F1: 49.51 | EM: 46.50 / F1: 51.32 | ✓ *(OOD)* | — | EM: 44.60 / F1: 47.57 | — | — | — | — *(WebQ)* | — |
| **10. LinearRAG** *(ICLR'26, Avg Contain/GPT)* | Avg-Acc: 65.40 | Avg-Acc: 66.95 | Avg-Acc: 35.45 | — | — | — | — | — | — | — |
| **Ref. Fact-Grounding Gap** *(EMNLP'26)* | ✓ | ✓ | Gate F1: 78.5 *(repro: 79.07)* | — | — | — | — | — | — *(có IIRC)* | — |

---

## 3. Tóm tắt Trade-off

- **Indexing-heavy / Inference-fast** (*HippoRAG 2, HopRAG, KET-RAG, LinearRAG*): Chi phí dồn vào bước dựng đồ thị (KET-RAG & LinearRAG giảm mạnh chi phí LLM indexing nhưng đánh đổi độ tinh vi quan hệ) để truy xuất multi-hop nhanh trong 1–2 bước.
- **Training/Inference-heavy** (*CoRAG, Search-R1, DeepResearcher, Search-o1, DeepRAG*): Giữ index nhẹ (BM25/E5/Web), đổ compute vào RL/SFT hoặc test-time scaling (`best-of-N`, long-CoT).
- **Lỗ hổng chung:** Hầu hết bài RAG 2025–2026 tối ưu truy xuất sâu hơn nhưng rất ít bài đánh giá **khả năng dừng/từ chối khi thiếu bằng chứng** (*IIRC, FaithEval, MultiHop-RAG `null_query`*).
