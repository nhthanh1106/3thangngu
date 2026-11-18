# Báo cáo model và benchmark KPR–BGE trên SciFact

_Cập nhật: 19/11/2026. Báo cáo này tổng hợp lượt chạy đã lưu trong repo; không phải kết quả của một lượt chạy mới._

## 1. Model đang dùng

| Vai trò | Checkpoint | Nguồn |
|---|---|---|
| Model cần đánh giá | `knowledgeable-ai/kpr-bge-base-en-v1.5` (KPR) | [Model card KPR](https://huggingface.co/knowledgeable-ai/kpr-bge-base-en-v1.5) |
| Model đối chứng | `BAAI/bge-base-en-v1.5` (BGE) | [Model card BGE](https://huggingface.co/BAAI/bge-base-en-v1.5) |

KPR là **Knowledgeable Passage Retriever** của bài [*Dynamic Injection of Entity Knowledge into Dense Retrievers*](https://aclanthology.org/2025.findings-emnlp.915/) (Yamada và cộng sự, Findings of EMNLP 2025). Checkpoint này dùng `bge-base-en-v1.5` làm model nền, bổ sung attention giữa ngữ cảnh và thực thể cùng embedding thực thể có thể cập nhật.

| Thông số KPR | Giá trị |
|---|---|
| Kích thước model công bố | 112 triệu tham số |
| Chiều embedding | 768 |
| Độ dài chuỗi tối đa | 512 token |
| Ngôn ngữ | Tiếng Anh |
| Revision trong notebook | `2cca6d26e56a02390b07247fc567afe34a46a184` |

**112 triệu tham số không tính bảng embedding thực thể**, vì model card cho biết bảng này nằm ở RAM CPU. Notebook ước tính các file bắt buộc để suy luận KPR cần tải khoảng **12,20 GB**; đây là dung lượng file, không phải VRAM. BGE đối chứng có embedding 768 chiều, giới hạn 512 token và revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`.

## 2. Cấu hình lượt chạy đã lưu

Nguồn: [notebook](../kpr-vs-bge-scifact-sampled-evaluation.ipynb), [README](../README.md), [báo cáo JSON](../kpr_eval_report.json).

| Mục | Giá trị |
|---|---|
| Tác vụ | Xếp hạng passage liên quan với truy vấn SciFact |
| Dữ liệu | BEIR SciFact test qrels |
| Lấy mẫu | 40 truy vấn, 800 passage ứng viên, seed 42 |
| Pool | 49 cặp query–passage liên quan, 44 passage liên quan duy nhất; còn lại là passage gây nhiễu |
| Suy luận | `cuda:0`, `float32`; batch size 8 cho query và passage |
| Query | KPR dùng query gốc; BGE thêm `Represent this sentence for searching relevant passages: ` |
| Xếp hạng | Chuẩn hóa L2 embedding rồi dùng tích vô hướng, tương đương cosine similarity |
| Vector đầu ra | Query `(40, 768)` và passage `(800, 768)` cho mỗi model |
| Trạng thái | Notebook ghi `success`; không huấn luyện hoặc fine-tune |

Đây là **restricted candidate retrieval**: mỗi truy vấn được xếp hạng trên 800 passage đã chọn, không phải toàn bộ corpus SciFact.

## 3. Kết quả đã lưu

| Metric | BGE | KPR | KPR − BGE |
|---|---:|---:|---:|
| Recall@1 | 0.6958 | 0.6708 | −0.0250 |
| Recall@5 | 0.8750 | 0.8250 | −0.0500 |
| Recall@10 | 0.9000 | 0.9000 | 0.0000 |
| MRR@10 | 0.8069 | 0.7778 | −0.0292 |
| nDCG@10 | 0.8312 | 0.8078 | −0.0233 |

Trong mẫu này, KPR bằng BGE ở Recall@10 và thấp hơn ở các metric còn lại. Chưa có khoảng tin cậy hoặc kiểm định thống kê cho các chênh lệch.

## 4. Mức độ tái tạo paper và hướng nghiên cứu nhóm

**Chưa tái tạo kết quả paper KPR.** Repo dùng checkpoint đã công bố để so sánh inference trên mẫu SciFact riêng; chưa tái hiện đầy đủ dữ liệu, pool, huấn luyện hoặc bảng kết quả của paper. Qrels SciFact ở đây đo mức liên quan của passage với claim, không gắn nhãn passage ủng hộ hay phản bác claim.

**Chưa tái tạo Fact-Grounding Gap.** Notebook hiện không có decision model, SRO, TRA hoặc uncertainty, và chưa thay câu hỏi thường bằng SRO để benchmark fact grounding. Không thể suy ra hiệu quả của hướng SRO–TRA–uncertainty từ kết quả retrieval KPR–BGE này.

## 5. Giới hạn và lưu ý khi chạy lại

- Chỉ 40 truy vấn và pool 800 passage; kết quả không đại diện cho full-corpus BEIR hoặc mọi tình huống entity retrieval.
- Notebook ghi cảnh báo lệch phiên bản Sentence Transformers: checkpoint KPR tạo với `5.2.0.dev0`, môi trường chạy dùng `5.1.1`. Có thêm cảnh báo tokenizer về `max_length` khi `padding=True` mà không có truncation strategy. Cần kiểm tra hai cảnh báo này khi chạy lại.
- Các con số trong mục 3 là **kết quả đã lưu trong repo**. Báo cáo này không chạy lại benchmark và không bổ sung metric mới.

## Tài liệu đối chiếu

1. Yamada và cộng sự, [*Dynamic Injection of Entity Knowledge into Dense Retrievers*](https://aclanthology.org/2025.findings-emnlp.915/), Findings of EMNLP 2025.
2. [KPR model card](https://huggingface.co/knowledgeable-ai/kpr-bge-base-en-v1.5).
3. [BGE model card](https://huggingface.co/BAAI/bge-base-en-v1.5).
4. File gốc: [notebook](../kpr-vs-bge-scifact-sampled-evaluation.ipynb), [JSON](../kpr_eval_report.json), [README](../README.md).
