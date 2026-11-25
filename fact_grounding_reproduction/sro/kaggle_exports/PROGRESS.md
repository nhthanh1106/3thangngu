# Kết quả nghiên cứu SRO theo từng phase

Cập nhật 25/11/2026. Báo cáo này ghi **việc đã thực hiện, kết quả, đánh giá và giới hạn** của từng phase. Những con số từ Kaggle được ghi theo lượt đã quan sát; không coi code chuẩn bị hoặc run đang chạy là kết quả hoàn tất.

## Phase 0 — Xác minh mốc classifier raw

**Đã làm:** Nạp checkpoint epoch 2 của Hùng và chạy inference lại trên toàn bộ 6.404 hop MuSiQue dev; đối chiếu dự đoán theo từng hàng với mốc stage 1.

**Kết quả:** [Verifier Kaggle v1](https://www.kaggle.com/code/hungluongduc/fact-grounding-raw-epoch-2-reload-verification?scriptVersionId=372055387) chạy thành công, F1 **0,7935989549**. [Notebook local](phase0/raw_epoch2_verifier_v1_372055387.ipynb) chỉ xác minh checkpoint, không train lại.

**Đánh giá:** Đây là mốc raw cũ để đối chiếu. Trọng số và dự đoán đầy đủ được giữ trên Kaggle.

## Phase 1 — Kiểm tra hợp đồng CSV

**Đã làm:** Xây parser, serializer và validator cho CSV `text,label`; kiểm tra 46.610 train và 6.404 dev, gồm số hàng, thứ tự, passage, nhãn và split.

**Kết quả:** Dry run và 6 test hợp đồng đạt; dữ liệu được bảo toàn qua chuyển dạng. Phase này chạy bằng script local, không có notebook Kaggle riêng.

**Đánh giá:** CSV gốc không có `qid`, nên không khôi phục chắc chắn câu hỏi QA cha từ từng hàng. Giới hạn này cũng ảnh hưởng cách chia validation ở phase 4.

## Phase 2 — Khóa schema và prompt SRO

**Đã làm:** Thử cách chuyển sub-question thành SRO trên **31 câu train**, audit kết quả rồi sửa một trường hợp phụ thuộc thực thể trước khi khóa correction v2.

**Kết quả:** 26 câu có trạng thái `ok`, 5 câu `ambiguous`, không có lỗi cấu trúc trong pilot. Phase này là pilot trong chat và audit local, không có notebook Kaggle riêng.

**Đánh giá:** Pilot chỉ xác nhận format và các lỗi nhận thấy trên mẫu train. Metadata model ID/snapshot của phiên chat không được host xác minh; không dùng dev để chỉnh prompt.

## Phase 3 — Tạo dữ liệu SRO

**Đã làm:** Sinh SRO cho **17.600 câu khác nhau**, rồi lắp lại CSV `text,label` cho train/dev. Audit ghép cặp raw–SRO và chạy kiểm tra độ dài bằng tokenizer checkpoint trên Kaggle.

**Kết quả:** Đủ **46.610 train/6.404 dev**; audit assembler đạt, thứ tự, passage và nhãn khớp nguồn. Có 2.270 hàng train và 537 hàng dev dùng fallback `ambiguous`. [Audit tokenizer v1](https://www.kaggle.com/code/hungluongduc/fact-grounding-sro-phase-3-token-length-audit?scriptVersionId=372156039) ghi raw **0/0** hàng vượt 512 token (train/dev), SRO **294/37**. [Notebook local](phase3/sro_token_length_audit_v1_372156039.ipynb) dùng để audit, không phải notebook sinh SRO.

**Đánh giá:** Số lượng và tính bảo toàn CSV đã đạt. Chất lượng ngữ nghĩa của SRO mới được kiểm theo mẫu, chưa bảo đảm mọi triple giữ đủ ý câu hỏi; các hàng quá 512 token có nguy cơ bị cắt khi train/inference.

## Phase 4 — So sánh classifier offline

**Đã làm:** Phase 4B đưa SRO vào checkpoint raw cũ để đo ảnh hưởng đổi định dạng. Phase 4C train lại **cả raw và SRO** từ cùng DeBERTa-v3-large, seed 42, split train/validation, cấu hình và quy tắc dừng tối đa 10 epoch/patience 3; chọn checkpoint bằng validation nội bộ rồi chấm paired một lần trên dev.

**Kết quả:**

| Phép thử | Raw F1 | SRO F1 | Ý nghĩa |
|---|---:|---:|---|
| [Probe 4B](https://www.kaggle.com/code/hungluongduc/fact-grounding-sro-phase-4b-checkpoint-probe?scriptVersionId=372160710) | 0,793599 | 0,518271 | Checkpoint cũ gặp đổi phân phối input; không phải so sánh công bằng sau retrain. |
| [Train lại 4C, dev 6.404 hàng](https://www.kaggle.com/code/hungluongduc/fact-grounding-sro-phase-4c-epoch-3-retry?scriptVersionId=373620770) | **0,788544** | **0,759722** | SRO thấp hơn raw **0,028822 F1**; 726 hàng có dự đoán khác nhau. |

Raw có best validation F1 **0,808704** ở step 2639 và dừng epoch 4; SRO có best validation F1 **0,782241** ở step 5278 và dừng epoch 5. [Audit CPU v6](https://www.kaggle.com/code/hungluongduc/fact-grounding-sro-phase-4c-epoch-3-retry?scriptVersionId=373721176) tính lại metric, hàng, nhãn, hash và threshold từ output paired, trạng thái `passed`. Dev không dùng để chọn checkpoint hoặc ngưỡng.

**Đánh giá:** Trong protocol classifier offline khi passage giữ nguyên, SRO **chưa cải thiện** khả năng phân loại đủ bằng chứng. Validation chỉ nhóm theo normalized sub-question vì CSV thiếu parent `qid`; 37 hàng dev SRO vượt 512 token và audit ý nghĩa chưa bao phủ toàn bộ. Kết quả classifier chưa tự động suy ra kết quả QA đầu cuối.

**Notebook local phase 4:** [4B probe](phase4b/raw_checkpoint_sro_probe_v1_372160710.ipynb). Tám bản phase 4C dưới đây được xếp theo thứ tự thực hiện:

| Thứ tự | Notebook | Chức năng và mốc |
|---|---|---|
| 01 | [Raw epoch 1](phase4c/01_raw_epoch1_training_profile_v1_372438728.ipynb) | Train raw, lưu checkpoint epoch 1. |
| 02 | [SRO epoch 1](phase4c/02_sro_epoch1_training_profile_v1_372438960.ipynb) | Train SRO cùng cấu hình, lưu checkpoint epoch 1. |
| 03 | [Cặp epoch 2](phase4c/03_paired_epoch2_training_v2_372571642.ipynb) | Resume hai nhánh, lưu checkpoint step 5278. |
| 04 | [Cặp epoch 3](phase4c/04_paired_epoch3_resume_v1_373104145.ipynb) | Resume hai nhánh, lưu checkpoint step 7917. |
| 05 | [Cặp epoch 4](phase4c/05_paired_epoch4_resume_v3_373276670.ipynb) | Resume sau khi sửa lỗi tải model; raw đạt patience 3. |
| 06 | [SRO epoch 5](phase4c/06_sro_epoch5_resume_v4_373428580.ipynb) | Chỉ SRO train tiếp và đạt patience 3. |
| 07 | [Inference dev](phase4c/07_paired_dev_inference_v5_373620770.ipynb) | Tạo dự đoán paired và metric 6.404 hàng. |
| 08 | [Audit dev](phase4c/08_paired_dev_audit_v6_373721176.ipynb) | Tính lại và kiểm tra output paired trên CPU. |

## Phase 5 — Retrieval và QA MuSiQue

**Đã làm:** Ghép **2.417 QA/6.404 hop** với câu hỏi raw/SRO; dựng corpus 21.100 passage và BM25/SQLite FTS5. Chạy bốn nhánh: A raw query/raw gate, B raw query/SRO gate, C SRO query/raw gate, D SRO query/SRO gate. Thử hai chế độ oracle normalization và predicted-substitution bằng Kaggle GPT-5.4-mini cho QA/judge.

**Kết quả retrieval:** Recall@10 ban đầu trên toàn dev là raw **5.599/6.404** và SRO **5.570/6.404 hop**. SRO thấp hơn 29 hop; đây là retrieval oracle, chưa phải QA accuracy.

**Kết quả QA đã chạy:**

| Chế độ và tập QA | A | B | C | D | Trạng thái |
|---|---:|---:|---:|---:|---|
| [Oracle pilot 30 QA](https://www.kaggle.com/code/chnglng/fact-grounding-sro-phase-5-paired-online?scriptVersionId=373748206), số câu đúng | 18 | 18 | 16 | 16 | Audit run `passed`. |
| [Predicted pilot 30 QA](https://www.kaggle.com/code/chnglng/fact-grounding-sro-phase-5-paired-online?scriptVersionId=373759167), số câu đúng | 12 | 12 | 13 | 14 | Audit run `passed`. |
| [Oracle QA 0–499](https://www.kaggle.com/code/chnglng/fact-grounding-sro-phase-5-paired-online?scriptVersionId=373759585), accuracy | 0,722 | 0,704 | 0,716 | 0,698 | 500/500; số đọc từ manifest log. |
| [Predicted QA 0–499](https://www.kaggle.com/code/chnglng/fact-grounding-sro-phase-5-paired-online?scriptVersionId=373779537), accuracy | 0,610 | 0,584 | 0,610 | 0,602 | 500/500; số đọc từ manifest log. |

Hai pilot có artifact `manifest.json` và `paired.jsonl` trên Kaggle. Hai lát 500 QA đã chạy xong, nhưng metric mới được đọc từ log, chưa audit độc lập từng dòng `paired.jsonl`. [Oracle QA 500–999 v7](https://www.kaggle.com/code/chnglng/fact-grounding-sro-phase-5-paired-online?scriptVersionId=373942559) đang chạy; mốc quan sát cuối lúc 20:21 ICT 25/11 là 92/500 QA, chưa có kết quả cuối.

**Đánh giá:** Trên pilot và lát đầu, chưa thấy nhánh SRO cải thiện ổn định so với A. Cỡ mẫu và phạm vi hiện tại chưa đủ kết luận toàn bộ 2.417 QA. Pipeline dùng SQLite FTS5/BM25 và GPT-5.4-mini, khác hạ tầng/model paper; predicted-substitution vẫn dùng template bắt nguồn từ normalization oracle. Cần hoàn tất các lát còn lại, kiểm evidence và chấm paired toàn dev trước khi kết luận phase 5.

**Notebook local phase 5** từ cùng notebook Kaggle của account `chnglng`:

| Thứ tự | Notebook | Việc thực hiện |
|---|---|---|
| 01 | [Oracle pilot v3](phase5/01_oracle_pilot_v3_373748206.ipynb) | Pilot 30 QA, oracle normalization. |
| 02 | [Predicted pilot v4](phase5/02_predicted_pilot_v4_373759167.ipynb) | Pilot 30 QA, predicted substitution. |
| 03 | [Oracle QA 0–499 v5](phase5/03_oracle_0000_0500_v5_373759585.ipynb) | Lát đầu 500 QA oracle. |
| 04 | [Predicted QA 0–499 v6](phase5/04_predicted_0000_0500_v6_373779537.ipynb) | Lát đầu 500 QA predicted. |

## Phase 6 — Benchmark mở rộng

**Trạng thái:** Chưa chạy HotpotQA, 2WikiMultiHopQA hoặc IIRC trong nhánh SRO; chưa có metric để báo cáo. Mỗi bộ cần protocol và scorer riêng trước khi đánh giá.
