# Tổng hợp đánh giá benchmark

Repo này lưu notebook, báo cáo kết quả và ghi chú tiếng Việt cho các phép đánh giá retrieval. Mỗi benchmark nên có notebook chạy được, file kết quả máy đọc được và README giải thích dữ liệu, cách chấm cùng giới hạn của kết luận.

## KPR và BGE trên BEIR SciFact

Đây là phép đánh giá retrieval nhẹ trên một mẫu của **BEIR SciFact**, dùng test split qrels. Nó so sánh checkpoint `knowledgeable-ai/kpr-bge-base-en-v1.5` với baseline `BAAI/bge-base-en-v1.5`.

**Phạm vi:** Notebook này chỉ chạy phép so sánh inference giữa hai checkpoint có sẵn. KPR thuộc paper *Dynamic Injection of Entity Knowledge into Dense Retrievers*, khác sáu paper nhóm trình bày (TRA-RAG, HEAR-RAG, SG-RAG, FACTA, CIRAG, Fact-Grounding Gap). Notebook KPR không benchmark Fact-Grounding Gap với SRO–TRA–uncertainty. Xem [báo cáo chi tiết về model, kết quả và mức độ tái tạo](reports/kpr-bge-scifact-model-benchmark.md).

### Dữ liệu và cách lấy mẫu

- Có 300 truy vấn SciFact đủ nhãn; notebook chọn 40 truy vấn bằng seed 42.
- Mỗi truy vấn được xếp hạng trên cùng một tập 800 passage ứng viên.
- Tập ứng viên giữ lại các passage có qrel dương của những truy vấn được chọn, rồi lấy thêm passage gây nhiễu để đủ 800. Mẫu có 49 cặp truy vấn–passage liên quan, tương ứng 44 passage liên quan duy nhất và 756 passage gây nhiễu.
- Đây là **restricted candidate pool**, không phải chạy toàn bộ corpus SciFact như một benchmark BEIR đầy đủ.
- Qrels đánh giá mức độ liên quan của bằng chứng với claim. Chúng không gắn nhãn passage có ủng hộ hay phản bác claim hay không.

### Cách đánh giá

Hai model mã hóa claim và passage thành vector. Notebook chuẩn hóa vector L2, dùng tích vô hướng (tương đương cosine similarity) để xếp hạng passage cho từng claim, rồi lấy trung bình metric trên 40 truy vấn:

- **Recall@1/5/10:** tỷ lệ passage liên quan được tìm thấy trong top k.
- **MRR@10:** thứ hạng của passage liên quan đầu tiên trong top 10; thứ hạng càng cao càng tốt.
- **nDCG@10:** chất lượng thứ tự xếp hạng trong top 10, có xét mức liên quan và vị trí.

Cả hai model chạy trên `cuda:0`, float32, embedding 768 chiều.

### Kết quả

| Metric | BGE | KPR | KPR − BGE |
|---|---:|---:|---:|
| Recall@1 | 0.6958 | 0.6708 | −0.0250 |
| Recall@5 | 0.8750 | 0.8250 | −0.0500 |
| Recall@10 | 0.9000 | 0.9000 | 0.0000 |
| MRR@10 | 0.8069 | 0.7778 | −0.0292 |
| nDCG@10 | 0.8312 | 0.8078 | −0.0233 |

Trong mẫu này, KPR bằng BGE ở Recall@10 nhưng thấp hơn ở các metric còn lại. Kết quả chưa đủ để kết luận model nào tốt hơn nói chung: chỉ có 40 truy vấn, tập ứng viên bị giới hạn và chưa có khoảng tin cậy hay kiểm định ý nghĩa thống kê. Thời gian mã hóa được ghi trong JSON cho một lượt chạy; không nên xem đây là benchmark latency ổn định.

### Các file

- [`kpr-vs-bge-scifact-sampled-evaluation.ipynb`](kpr-vs-bge-scifact-sampled-evaluation.ipynb): notebook Kaggle gồm mã chuẩn bị môi trường, tải và kiểm tra model, lấy mẫu SciFact, tính metric và xuất báo cáo. Các output trong notebook lưu log và kết quả của lượt chạy.
- [`kpr_eval_report.json`](kpr_eval_report.json): báo cáo có cấu trúc gồm cấu hình, revision model, thông tin dữ liệu và môi trường, metric tổng hợp, thời gian, kết quả theo từng truy vấn và phân tích các trường hợp hai model xếp hạng khác nhau. Kiểm tra `execution.status`, `execution.errors` và `execution.warnings` để biết trạng thái chạy.
- `README.md`: phần giới thiệu benchmark, giải thích từng file, metric và giới hạn diễn giải.

## Giới hạn và bước tiếp theo

Đây là lần kiểm tra ban đầu trên mẫu có kiểm soát, không phải tái lập đầy đủ kết quả của BEIR hay bài báo KPR. Muốn kết luận chắc hơn, cần chạy toàn bộ test queries trên toàn bộ corpus ứng viên, báo cáo khoảng tin cậy bằng so sánh paired theo truy vấn, và lặp lại đo thời gian.

## Reproduction fact-grounding classifier

Notebook, checkpoint DeBERTa-v3-large và hướng dẫn tái lập bài fact-grounding được lưu trong [`fact_grounding_reproduction/`](fact_grounding_reproduction/). Xem [README hướng dẫn](fact_grounding_reproduction/README.md) để biết dữ liệu, kết quả, giới hạn và cách lấy checkpoint qua Git LFS.

Một lượt Kaggle **riêng** trên tài khoản `hungluongduc` đã chạy đủ ba epoch và đạt MuSiQue dev F1 = 0,7936 ở checkpoint epoch 2, so với 0,785 trong paper. Xem [báo cáo kết quả và các khác biệt về protocol](reports/fact-grounding-gap-deberta-musique-kaggle-results.md). Checkpoint của lượt này nằm ở output Kaggle stage 1, không phải checkpoint trong `fact_grounding_reproduction/`.
