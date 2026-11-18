# Báo cáo model và benchmark KPR–BGE trên SciFact

_Cập nhật: 19/11/2026. Báo cáo này tổng hợp lượt chạy đã lưu trong repo; không phải kết quả của một lượt chạy mới._

> **Trạng thái nghiên cứu:** Notebook hiện chỉ chạy **mã đánh giá so sánh hai checkpoint retrieval có sẵn (KPR và BGE)** trên một mẫu SciFact. KPR xuất phát từ một paper **khác sáu paper nhóm trình bày**. Kết quả này chưa phải reproduction của paper KPR, cũng chưa phải baseline hoặc reproduction của bất kỳ paper nào trong sáu paper đó. Đặc biệt, chưa có benchmark Fact-Grounding Gap với SRO, TRA và uncertainty.

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

## 4. Notebook đã làm gì và mức độ tái tạo paper

### 4.1. Phạm vi đã chạy thực tế

1. Tải và kiểm tra mã nguồn/checkpoint KPR được công bố; tải checkpoint BGE để đối chứng.
2. Lấy mẫu 40 truy vấn SciFact và lập cùng một pool 800 passage cho hai model.
3. Mã hóa truy vấn và passage thành embedding, xếp hạng passage bằng similarity.
4. Tính Recall@1/5/10, MRR@10, nDCG@10; lưu output trong notebook và xuất báo cáo JSON.

Đây là **mã benchmark so sánh model do notebook thiết lập**, chạy inference trên checkpoint có sẵn. Notebook không chứng minh đã chạy lại mã huấn luyện hay protocol đánh giá đầy đủ của tác giả KPR. Vì vậy, dùng cụm “đã chạy so sánh KPR–BGE trên mẫu SciFact” là chính xác hơn “đã reproduce paper KPR”.

### 4.2. Quan hệ với các paper nhóm trình bày

Paper cung cấp KPR là [*Dynamic Injection of Entity Knowledge into Dense Retrievers*](https://aclanthology.org/2025.findings-emnlp.915/), **nằm ngoài** sáu paper nhóm đang trình bày. Trong repo này, không có mã hoặc kết quả benchmark tái tạo các phương pháp sau:

| Paper trong phần trình bày | Bằng chứng triển khai trong notebook hiện tại | Trạng thái từ repo này |
|---|---|---|
| TRA-RAG | Không có module biểu diễn/khai thác TRA | Chưa reproduce |
| HEAR-RAG | Không có pipeline HEAR-RAG | Chưa reproduce |
| SG-RAG | Không có pipeline SG-RAG | Chưa reproduce |
| FACTA | Không có pipeline FACTA | Chưa reproduce |
| CIRAG | Không có pipeline CIRAG | Chưa reproduce |
| Fact-Grounding Gap | Không có phép đo fact grounding theo protocol của paper | Chưa reproduce |

Bảng này chỉ đánh giá **nội dung hiện có trong repo này**; không kết luận nhóm chưa chạy các paper đó ở repo, nhánh hoặc máy khác.

### 4.3. Vì sao kết quả hiện tại chưa trả lời câu hỏi nghiên cứu của nhóm

SciFact qrels trong notebook đo **passage có liên quan tới claim hay không**, không gắn nhãn passage ủng hộ/phản bác hoặc fact đã được grounding hay chưa. Đầu vào là truy vấn SciFact thông thường; notebook chưa chuyển truy vấn thành bộ **subject–relation–object (SRO)**, chưa xây biểu diễn **TRA**, chưa kết hợp hai biểu diễn bằng **decision model**, và chưa tính **uncertainty** để quyết định. Do đó các metric retrieval ở mục 3 không đo được hiệu quả của hướng SRO–TRA–uncertainty, cũng không phải kết quả benchmark Fact-Grounding Gap sau khi thay câu hỏi bằng SRO.

Để gọi một bước là reproduction Fact-Grounding Gap, cần đối chiếu dataset, input/output, nhãn, protocol và metric với paper gốc, chạy baseline theo protocol đó rồi mới thử biến thể SRO. Các bước này **chưa được thể hiện trong repo hiện tại**.

## 5. Giới hạn và lưu ý khi chạy lại

- Chỉ 40 truy vấn và pool 800 passage; kết quả không đại diện cho full-corpus BEIR hoặc mọi tình huống entity retrieval.
- Notebook ghi cảnh báo lệch phiên bản Sentence Transformers: checkpoint KPR tạo với `5.2.0.dev0`, môi trường chạy dùng `5.1.1`. Có thêm cảnh báo tokenizer về `max_length` khi `padding=True` mà không có truncation strategy. Cần kiểm tra hai cảnh báo này khi chạy lại.
- Các con số trong mục 3 là **kết quả đã lưu trong repo**. Báo cáo này không chạy lại benchmark và không bổ sung metric mới.

## Tài liệu đối chiếu

1. Yamada và cộng sự, [*Dynamic Injection of Entity Knowledge into Dense Retrievers*](https://aclanthology.org/2025.findings-emnlp.915/), Findings of EMNLP 2025.
2. [KPR model card](https://huggingface.co/knowledgeable-ai/kpr-bge-base-en-v1.5).
3. [BGE model card](https://huggingface.co/BAAI/bge-base-en-v1.5).
4. File gốc: [notebook](../kpr-vs-bge-scifact-sampled-evaluation.ipynb), [JSON](../kpr_eval_report.json), [README](../README.md).
