# 5 Nút Thắt Thực Nghiệm & Bản Đồ Khoảng Trống Nghiên Cứu (2024–2026)

Tài liệu này giải thích trực quan **5 nút thắt cốt lõi** rút ra từ đợt đánh giá 8 benchmark (xem [`eight_benchmark_sota_comparison.md`](eight_benchmark_sota_comparison.md)) và đối chiếu với các công trình 2024–2026 liên quan trực tiếp.

---

## 1. Giải thích 5 nút thắt cốt lõi

1. **Format Shift (Đổi dạng câu hỏi: câu tách nhỏ vs câu gộp nhiều bước)**
   - Model được huấn luyện chấm từng **sub-question** đơn lẻ nên khi giữ đúng format câu hỏi con (`02b_musique_full_hops`) hoặc chuyển hẳn sang domain tin tức (`07_multihop_rag_news`), nó đều đạt **AUROC ~0.89–0.90**.
   - Khi ném thẳng **câu hỏi multi-hop gộp chưa phân rã** (`02a_musique_full_q`), AUROC tụt ngay **−12.3 điểm** (`0.899 → 0.776`).
   - *Hàm ý:* Khi tích hợp vào RAG, bắt buộc giữ bước phân rã sub-question trước khi qua cổng DeBERTa.

2. **Cắt cụt ngữ cảnh (Truncation `1,500` ký tự / `512` token)**
   - Trên `05_2wiki_support_pairs`, `95.9%` mẫu vượt `1,500` ký tự và bị cắt mất phần đuôi chứa supporting sentence, kéo AUROC xuống `0.601`. Tương tự `06_faitheval_pairs` bị cắt `50.9%`.
   - *Hàm ý:* Cần một bước tỉa câu thừa (sentence pruning / reranking) hoặc chuyển sang encoder ngữ cảnh dài trước khi chấm sufficiency.

3. **Bẫy trùng từ khóa (Adversarial Lexical Overlap)**
   - Trên `08_squad_v2_dev` và `04_hotpotqa_pairs`, các mẫu âm tính cố tình chứa đầy từ khóa/thực thể giống câu hỏi nhưng thiếu quan hệ trả lời được.
   - Dù thứ hạng tương đối vẫn tốt (`AUROC 0.841` SQuAD2, `Pairwise Acc 88.55%` HotpotQA), ngưỡng cứng `0.5` bị lừa đoán dương hàng loạt (`FP = 4,076` SQuAD2, `2,295` HotpotQA).

4. **Bệnh "quá tự tin" & Hiệu chỉnh xác suất (Calibration — ECE & Temperature Scaling)**
   - Xác suất thô của DeBERTa fine-tuned bị đẩy sát `0` hoặc `1` (`ECE` cao).
   - Fit duy nhất 1 tham số **Temperature Scaling (`T = 1.9290`)** trên `01_musique_dev` (0 rò rỉ target test) đã cắt giảm **20%–68% ECE** trên 8/9 track (`01`: `0.118 → 0.037`, `02a`: `0.195 → 0.085`, `03`: `0.140 → 0.078`).

5. **Đánh đổi với LLM Judge & Kiến trúc Cổng 2 tầng (Cascade)**
   - Frontier LLM Judge (`Gemini 1.5 Pro`, `FLAMe-24B`, `GPT-4o`) chính xác hơn ~6–11 điểm ở các ca bẫy khó, nhưng chậm gấp 30–50× và tốn chi phí API từng hop.
   - DeBERTa 435M chạy `14–33 ms/mẫu` trong `~5 GiB VRAM` với chi phí `$0` → phù hợp làm **cổng tầng 1 (Tier-1 fast gate)** sau Temperature Scaling: tự chốt ở vùng chắc chắn (`p < 0.3` hoặc `p > 0.7`) và chỉ đẩy vùng xám (`0.3 ≤ p ≤ 0.7`) lên LLM Judge.

---

## 2. Bản đồ Paper 2024–2026 giải quyết từng nút thắt

| Nút thắt phát hiện | Paper liên quan trực tiếp | Venue / Thời điểm | Hướng tiếp cận chính |
|---|---|---|---|
| **Gap 1: Format Shift** | *Verification Without Sufficiency: Per-Chunk Filtering Fails on Multi-Hop RAG, and Decomposition Repairs It* | arXiv (08/2026) | Chứng minh classifier sufficiency thất bại trên câu multi-hop thô và phục hồi mạnh khi điều kiện hóa theo câu hỏi con đã phân rã. |
| **Gap 2: Cắt cụt ngữ cảnh** | **Provence** (*Efficient and Robust Context Pruning for RAG*)<br>*Knowing When to Stop* | **ICLR 2025** (A*)<br>arXiv (2026) | Gắn nhãn từng câu để cắt tỉa câu thừa trước khi đưa vào verifier/LLM; đọc tăng dần theo tín hiệu latent sufficiency. |
| **Gap 3: Bẫy trùng từ khóa** | **Sufficient Context** (Google Research)<br>**MiniCheck / Bespoke-MiniCheck** | **ICLR 2025** (A*)<br>**EMNLP 2024** (A*) | Phân biệt rạch ròi *Relevance* vs *Sufficiency*; huấn luyện verifier nhỏ bằng hard negatives tổng hợp tinh vi đạt chất lượng ngang GPT-4 với chi phí rẻ ~400×. |
| **Gap 4: Calibration & Abstention** | **SURE-RAG** · **UR-RAG** · **CalibJudge** | 2026 | Kết hợp đánh giá đủ bằng chứng với hiệu chỉnh bất định / conformal risk để quyết định Answer vs Abstain an toàn khi dịch chuyển miền. |
| **Gap 5: Cổng 2 tầng (Cascade)** | **UCCI** (*Calibrated Uncertainty for Cost-Optimal LLM Cascade Routing*)<br>**BEST-Route** | arXiv (05/2026)<br>**ICML 2025** (A*) | Chứng minh đặt ngưỡng chuyển tầng trên xác suất đã calibrated là tối ưu chi phí; dùng họ DeBERTa nhỏ làm router trước LLM lớn. |

---

## 3. Hướng mở rộng đề xuất (Kết hợp 3 mảnh chưa ai gộp trọn vẹn)

1. **Sentence Pruning trước khi chấm** (dùng `Provence` / sentence reranker rút gọn tài liệu dài xuống `< 512` tokens → xử lý Gap 2 trên 2Wiki / FaithEval / FRAMES).
2. **Hard-Negative Fine-Tuning v2** cho DeBERTa (trộn thêm ~10–15% hard negatives giàu lexical overlap từ SQuAD 2.0 + FaithEval + MuSiQue-Full unanswerable → xử lý Gap 3).
3. **Calibrated Tier-1/Tier-2 Cascade Gate** trong vòng lặp multi-hop re-retrieval (dùng Temperature Scaling `T = 1.929` giữ lại ~75–80% hop tự quyết ở DeBERTa và chỉ chuyển 20–25% ca vùng xám lên LLM Judge → xử lý Gap 4 & 5).
