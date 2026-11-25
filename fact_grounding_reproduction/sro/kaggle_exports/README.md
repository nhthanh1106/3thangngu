# Báo cáo nghiên cứu Fact Grounding với SRO

## Mục tiêu

Nghiên cứu kiểm tra việc chuyển sub-question thành các bộ ba **subject–relation–object (SRO)** có giúp xác định passage đã đủ bằng chứng để trả lời và cải thiện retrieval/QA hay không. Thí nghiệm hiện dùng MuSiQue; các benchmark khác chưa chạy.

[PROGRESS.md](PROGRESS.md) trình bày chi tiết từng phase: việc đã làm, kết quả, đánh giá và giới hạn. Năm thư mục phase bên dưới chứa **15 notebook đã chạy** được tải từ hai tài khoản Kaggle, đặt theo thứ tự thực hiện. Đây là bộ **báo cáo kết quả hiện có**, không phải gói tái tạo toàn bộ dữ liệu và checkpoint.

## Kết quả chính

| Nội dung | Kết quả và nhận xét |
|---|---|
| Dữ liệu | Đã tạo đủ 46.610 hàng train và 6.404 hàng dev dạng SRO; kiểm tra thứ tự, passage và nhãn khớp CSV raw. Audit ý nghĩa của từng SRO mới thực hiện theo mẫu. |
| Độ dài input | Với ngưỡng 512 token, raw không có hàng vượt ngưỡng; SRO có 294 hàng train và 37 hàng dev vượt ngưỡng. |
| Classifier offline | Train lại raw/SRO cùng protocol, chấm paired đủ 6.404 dev: F1 raw **0,788544**, SRO **0,759722**. Trong phép thử này SRO thấp hơn raw **0,028822 F1**. |
| Retrieval MuSiQue | BM25 recall@10 toàn dev: raw **5.599/6.404 hop**, SRO **5.570/6.404 hop**. Đây là kết quả retrieval oracle ban đầu, chưa phải QA. |
| QA MuSiQue | Bốn nhánh query/gate đã chạy pilot 30 QA và lát đầu 500 QA ở cả hai chế độ oracle/predicted-substitution. Chưa có kết quả gộp 2.417 QA; phase này đang tiếp tục. |

**Nhận xét hiện tại:** SRO chưa cho thấy lợi ích ở classifier offline hoặc recall retrieval ban đầu. Các lượt QA đã chạy còn là pilot/lát đầu và dùng pipeline biến thể so với paper, nên chưa kết luận về lợi ích đầu cuối trên toàn MuSiQue.

## Nguồn notebook Kaggle

- Account `hungluongduc`: [xác minh raw](https://www.kaggle.com/code/hungluongduc/fact-grounding-raw-epoch-2-reload-verification?scriptVersionId=372055387), [audit token SRO](https://www.kaggle.com/code/hungluongduc/fact-grounding-sro-phase-3-token-length-audit?scriptVersionId=372156039), [probe checkpoint](https://www.kaggle.com/code/hungluongduc/fact-grounding-sro-phase-4b-checkpoint-probe?scriptVersionId=372160710) và các lượt train/inference/audit phase 4C được liệt kê trong [PROGRESS.md](PROGRESS.md).
- Account `chnglng`: [notebook chung phase 5](https://www.kaggle.com/code/chnglng/fact-grounding-sro-phase-5-paired-online), với các bản v3–v6 đã hoàn tất và được lưu local. Bản v7 đang chạy ở mốc kiểm gần nhất.

Notebook và link Kaggle là bằng chứng cho các lượt đã chạy. File dự đoán từng hàng, checkpoint và dữ liệu nguồn không nằm trong bộ báo cáo này.
