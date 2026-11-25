# Kết quả tái tạo bộ phân loại Fact-Grounding trên MuSiQue

**Cập nhật:** 20/11/2026. **Trạng thái:** đã chạy xong ba epoch trên Kaggle và đánh giá trên toàn bộ dev. Đây là một lượt chạy riêng với bộ tái tạo trong [`fact_grounding_reproduction/`](../fact_grounding_reproduction/README.md).

## Phạm vi và nguồn

Lượt này fine-tune `microsoft/deberta-v3-large` từ hai CSV công khai của [repo tác giả](https://github.com/RichardZhu123/fact-grounding-gap), rồi chấm bộ phân loại đủ bằng chứng ở từng hop trên MuSiQue dev. Đối chiếu với [paper *Diagnosing the Fact-Grounding Gap in Multi-Hop Question Answering*](https://aclanthology.org/2026.emnlp-main.834/), §5, Bảng 4. Đầu vào là sub-question cùng các passage; nhãn 0 là NOT-ANSWERABLE, nhãn 1 là ANSWERABLE. Lượt fine-tune này không gọi GPT API vì nhãn đã nằm trong CSV công khai.

Hai Kaggle run nối tiếp nhau:

1. [Stage 1, version 371432426](https://www.kaggle.com/code/hungluongduc/fact-grounding-gap-deberta-full-musique?scriptVersionId=371432426): chạy epoch 1–2, lưu `checkpoint-5828` và `best_model_after_2_epochs`.
2. [Stage 2, version 371639287](https://www.kaggle.com/code/hungluongduc/fact-grounding-gap-deberta-resume-epoch-3?scriptVersionId=371639287): nạp `checkpoint-5828` cùng trạng thái optimizer/scheduler, chạy epoch 3 đến `global_step=8742`, lưu `checkpoint-8742` và `stage2_results.json`. JSON chọn checkpoint epoch 2 theo dev F1.

Đây không phải notebook KPR–BGE trên SciFact ở gốc repo. Repo còn có [một bộ tái tạo Fact-Grounding khác](../fact_grounding_reproduction/README.md) dùng tài khoản Kaggle `huthnhnguyn`; không gộp kết quả và checkpoint của hai lượt chạy.

## Dữ liệu và cấu hình đã chạy

| Mục | Giá trị |
|---|---|
| Train | 46.610 hàng; 25.125 nhãn 1, 21.485 nhãn 0 |
| Dev | 6.404 hàng; 3.145 nhãn 1, 3.259 nhãn 0 |
| SHA-256 train CSV | `7832387e08d082e4e2ca8b4161aefea2cc56450e003e14e312c3bc5f31c0557d` |
| SHA-256 dev CSV | `f7ec0b2308ebdc811bfc11828a076f498fd2cb542c503274dad5cba852ebd1bc` |
| Model gốc | `microsoft/deberta-v3-large` |
| Seed; độ dài tối đa | 42; 512 token |
| Batch hiệu dụng; learning rate | 16; `2e-5` |
| Warmup; weight decay | 875 optimizer steps; 0,01 |
| Thiết bị; precision | 2 × Tesla T4; FP16 |
| Điều chỉnh bộ nhớ | Dynamic padding; gradient checkpointing |

Lượt chạy dùng toàn bộ train/dev CSV, không lấy mẫu. Stage 1 mất khoảng 8 giờ 09 phút; stage 2 khoảng 3 giờ 39 phút. Cả hai đều kết thúc và ghi kết quả. Các số dưới đây là **dev**, không phải test độc lập.

## Kết quả thực tế và đối chiếu paper

| Metric | Paper, §5 Bảng 4 | Epoch 2, **được chọn** | Epoch 3 | Epoch 2 trừ paper |
|---|---:|---:|---:|---:|
| Precision | 0,805 | 0,8157 | 0,8173 | +0,0107 |
| Recall | 0,765 | 0,7727 | 0,7638 | +0,0077 |
| F1 | 0,785 | **0,7936** | 0,7896 | **+0,0086** |
| Accuracy | xem lưu ý bên dưới | 0,8026 | 0,8001 | — |

Metric epoch 2 chính xác từ log: precision `0.8157099698`, recall `0.7726550079`, F1 `0.7935989549`, accuracy `0.8026233604`. Epoch 3: precision `0.8172847907`, recall `0.7637519873`, F1 `0.7896120973`, accuracy `0.8001249219`. Dev F1 epoch 3 thấp hơn epoch 2 khoảng 0,0040; theo tiêu chí dev F1, checkpoint epoch 2 là kết quả được chọn.

Paper còn báo accuracy 79,4% ở **Bảng 5**, trong phần phân tích nhãn/đầu vào. Bảng đó không phải cùng hàng precision–recall–F1 ở Bảng 4; vì vậy không xem chênh lệch accuracy với 79,4% là phép đối chiếu trực tiếp đã xác nhận.

**Lưu ý metric:** trường `eval_flag_rate` do notebook in ra đếm `prediction == 1`, tức tỷ lệ dự đoán ANSWERABLE. Tên trường dễ gây nhầm với tỷ lệ *NOT-ANSWERABLE flag* của paper; không dùng trường này để so sánh trực tiếp.

## Mức độ tái tạo và giới hạn

Lượt chạy tái tạo **bài toán phân loại và điểm dev gần paper**, chưa tái tạo chính xác toàn bộ quy trình. [Script train chính thức](https://github.com/RichardZhu123/fact-grounding-gap/blob/main/train_deberta.py) đặt tối đa 10 epoch, warmup ratio 10%, early stopping patience 3 và chọn checkpoint bằng dev F1. Lượt Kaggle đặt kế hoạch ba epoch, warmup cố định 875 steps, dừng/sang notebook khác sau epoch 2, dùng dynamic padding và gradient checkpointing trên T4; phiên bản thư viện và phần cứng cũng khác. Những khác biệt này có thể thay đổi quỹ đạo learning rate và kết quả. Chưa có đánh giá trên test set độc lập.

Chưa chạy các phần tạo trajectory, GPT judge, đánh giá QA đầu cuối, can thiệp retrieval hoặc thử SRO–TRA–uncertainty. Vì vậy F1 0,7936 chỉ cho thấy bộ phân loại ở lượt này đạt điểm dev gần số paper, không chứng minh toàn bộ kết quả Fact-Grounding Gap hay cải thiện QA.

**Artifact cần giữ:** checkpoint tốt nhất nằm trong output stage 1 tại `fg_full_stage1/best_model_after_2_epochs`; output stage 2 chứa checkpoint epoch 3 và `fg_full_stage2/stage2_results.json`. Chỉ tải output stage 2 sẽ không có checkpoint được chọn. Repo nhóm hiện lưu báo cáo này; hai notebook và artifact của lượt chạy `hungluongduc` nằm ở các liên kết Kaggle phía trên.
