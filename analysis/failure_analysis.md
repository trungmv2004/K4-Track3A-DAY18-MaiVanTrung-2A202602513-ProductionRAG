# Failure Analysis — Lab 18: Production RAG

**Họ và tên:** Mai Văn Trung

**MSSV:** 2A202602513

**Ngày kiểm tra:** 04/10/2026 (GMT+7)

## Phạm vi và bằng chứng

Gemini 3.5 Flash-Lite hết quota ngày: probe chat trả HTTP 429 với limit 500
request/ngày; embedding vẫn hoạt động. Người dùng chọn model khác còn quota,
`.env` đổi sang `gemini-3.1-flash-lite`. Script `rerun_gemini_evaluation.py`
đã chấm đủ 20/20 điểm hữu hạn cho mỗi metric, cả baseline và production,
cùng evaluator Gemini 3.1 và embedding `gemini-embedding-001`.

Baseline giữ 20 câu trả lời đã tạo bằng Gemini 3.5 và chấm lại toàn bộ bằng
Gemini 3.1; production tạo mới đủ 20 câu bằng Gemini 3.1, không fallback.
Model generation khác nhau nên chênh lệch không chỉ đo tác động của retrieval/
reranking; không khẳng định pipeline là nguyên nhân duy nhất của chênh lệch.

Retrieval dùng MiniLM-L6-v2, reranker TinyBERT-L2-v2 theo lựa chọn model nhẹ.
26 tài liệu text tạo 26 parents và 101 children; hai PDF scan chưa OCR không
được index. Dùng lại 101 payload enrichment Gemini 3.5 đã kiểm tra source/
parent/metadata. Timing enrichment là kiểm tra cache, không phải 101 API calls.

Hai ô baseline câu 16 lỗi kết nối được chấm bổ sung, giữ mọi điểm hữu hạn;
production chấm đủ không cần retry metric. Checkpoint gắn đúng question, answer,
context, ground truth và evaluator bằng fingerprint. Không gán điểm cũ vào
inputs của lần lỗi. Điểm không được làm tròn hoặc chọn lại để tăng điểm.

Nguồn: hai báo cáo RAGAS, `reports/gemini_quota_diagnostic.json` và
`reports/diagnostic_review.json`. Giữ báo cáo lỗi ở
`reports/ragas_failed_quota_report.json`; điểm cũ với đúng inputs ở
`reports/ragas_completed_checkpoint_report.json`. Trace candidates/reranking
được replay local trên index khôi phục và đối chiếu các parents được chọn.

## RAGAS Scores

| Metric | Baseline chấm lại | Production | Δ |
|---|---:|---:|---:|
| faithfulness | 0,9679 | 0,9317 | -0,0362 |
| answer_relevancy | 0,5075 | 0,9009 | 0,3933 |
| context_precision | 0,3583 | 0,6833 | 0,3250 |
| context_recall | 0,4333 | 0,8000 | 0,3667 |

Production có ba metrics ≥ 0,70: Faithfulness, Relevancy, Recall. Precision
0,6833 chưa đạt 0,70. Đủ điều kiện 10 điểm tiêu chí RAGAS #7 và bonus
Faithfulness ≥ 0,85; chưa đủ bonus tất cả metrics ≥ 0,75. Faithfulness cao
không bảo đảm answer đúng/đủ ý, như câu 4.

## Bottom-5 theo RAGAS thật

Sắp theo trung bình số học bốn metrics tăng dần; hòa điểm giữ thứ tự test set.
Không dùng token overlap hoặc synthetic probe làm điểm.

| Thứ tự | Câu | Faithfulness | Relevancy | Precision | Recall | Trung bình |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 1,0000 | 0,0000 | 0,0000 | 0,0000 | 0,2500 |
| 2 | 12 | 1,0000 | 0,9088 | 0,0000 | 0,5000 | 0,6022 |
| 3 | 5 | 1,0000 | 0,9378 | 0,0000 | 0,5000 | 0,6094 |
| 4 | 17 | 0,3333 | 0,9412 | 1,0000 | 0,5000 | 0,6936 |
| 5 | 7 | 1,0000 | 0,9278 | 0,5000 | 0,5000 | 0,7320 |

### #1 — Câu 4: Số ngày phép năm

- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** Hiện hành 15 ngày; bản cũ 12 ngày đã bị thay thế.
- **Got:** Nhân viên thử việc không được phép năm; không tìm thấy tổng số ngày phép cho nhân viên chính thức.
- **Contexts:** `thu_viec.md` → `nghi_phep_khong_luong.md` → `nghi_om.md`.
- **Worst metric:** answer_relevancy = 0,0000.
- **Error Tree:** Output có 15 ngày? Không → Context có chính sách phép năm hiện hành? Không → Query rõ? Có → Candidates có nguồn đúng? Có → M3/top-3 làm mất nguồn.
- **Diagnosis:** `nghi_phep_nam_v2024.md` có trong candidates nhưng đứng thứ 5 theo parents khác nhau sau reranking. Đây là lỗi chọn context sau ranking, không phải tài liệu chưa được index. Generator bám ba nguồn sai chủ đề nên không bịa số, nhưng không trả lời được câu hỏi.
- **Fix:** Đo reranker BGE theo cấu hình đề bài và Coverage@3 sau ranking. Đo candidate recall riêng để không sửa nhầm chunking/retrieval; không hardcode 15 ngày theo test set.

### #2 — Câu 12: Senior, chín năm thâm niên và lương

- **Question:** Senior có chín năm thâm niên được nghỉ bao nhiêu ngày phép và lương khoảng nào?
- **Expected:** 15 + 9/3 = 18 ngày phép; lương Senior 20–35 triệu VNĐ/tháng.
- **Got:** Đúng 18 ngày phép [2], nhưng không tìm thấy mức lương cụ thể trong context.
- **Contexts:** `nghi_phep_khong_luong.md` → `nghi_phep_nam_v2024.md` → `nghi_om.md`.
- **Worst metric:** context_precision = 0,0000.
- **Error Tree:** Output đủ hai phần? Không → Context có bảng lương Senior? Không → Query rõ, nhiều ý? Có → Candidates có bảng lương? Có → Ranking/budget mất nguồn cho ý thứ hai.
- **Diagnosis:** `bang_luong_2024.md` có trong candidates nhưng đứng thứ 9 theo parents khác nhau sau reranking. Trace xác định mất coverage ở xếp hạng/chọn context, thay vì chỉ suy đoán retrieval thiếu nguồn. Generator thừa nhận thiếu thông tin thay vì bịa mức lương.
- **Fix:** Tách ý phép năm và lương thành subqueries, hợp nhất candidates rồi giữ coverage từng ý trong budget. Đo source coverage trước/sau ranking trên holdout đa nguồn; không hardcode tên file hay mức lương Senior.

- **Kiểm tra judge:** Precision được chấm 0 dù nguồn hiện hành có ích cho một phần câu hỏi. Không diễn giải 0 thành mọi context đều vô dụng; cần lưu verdict để kiểm tra ground truth nhiều ý/phạm vi phiên bản.

### #3 — Câu 5: Thâm niên cộng ngày phép

- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** Từ ba năm, cộng một ngày mỗi ba năm; ground truth còn so sánh bản cũ yêu cầu năm năm.
- **Got:** Trả lời đúng từ 3 năm trở lên được cộng ngày phép [3].
- **Contexts:** `nghi_phep_khong_luong.md` → `mentor_buddy.md` → `nghi_phep_nam_v2024.md`.
- **Worst metric:** context_precision = 0,0000.
- **Error Tree:** Output đúng hiện hành? Có → Context liên quan đứng đầu? Không → Context có điều kiện năm năm của bản cũ? Không → Query yêu cầu lịch sử? Không → Ranking nhiễu và phạm vi ground truth.
- **Diagnosis:** Gemini đọc nguồn thứ ba nên answer đúng. Hai parents đầu không trả lời ngày phép theo thâm niên. Recall thấp còn gắn với phần bản cũ trong ground truth; pipeline mặc định loại superseded, và bản hiện hành không chứa điều kiện năm năm của bản cũ.
- **Fix:** Cải thiện ranking để nguồn phép năm lên trước. Trên holdout, tách câu hỏi hiện hành và so sánh lịch sử; chỉ lấy bản cũ có nhãn superseded khi query cần. Giữ nguyên test set và điểm này, không sửa ground truth để tăng điểm.

- **Kiểm tra judge:** Precision được chấm 0 dù nguồn hiện hành có ích cho một phần câu hỏi. Không diễn giải 0 thành mọi context đều vô dụng; cần lưu verdict để kiểm tra ground truth nhiều ý/phạm vi phiên bản.

### #4 — Câu 17: Phí phạt tạm ứng 15 triệu sau 20 ngày

- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn 15 ngày, quá hạn 5 ngày; 300.000 VNĐ/tháng; ground truth còn yêu cầu pro-rata khoảng 50.000 VNĐ cho 5 ngày.
- **Got:** Nêu thời hạn 15 ngày, quá hạn 5 ngày và phí 2%/tháng trên 15 triệu; chưa tính rõ 300.000 VNĐ/tháng.
- **Contexts:** `tam_ung.md` → `nghi_phep_nam_v2024.md` → `nghi_phep_khong_luong.md`.
- **Worst metric:** faithfulness = 0,3333.
- **Error Tree:** Output ra số tiền? Chưa → Context có rate 2%/tháng? Có → Có quy ước chia ngày? Không → Judge nhận dữ kiện query/phép tính? Chưa xác minh → Generation tính toán + kiểm tra judge/ground truth.
- **Diagnosis:** Nguồn đúng có và đứng đầu. Không kết luận hallucination chỉ từ Faithfulness thấp: 15 triệu và 20 ngày đến từ query; 5 ngày là phép trừ. Chưa lưu NLI verdict để xác định statement bị bác bỏ. Ground truth 50.000 cho 5 ngày giả định pro-rata/tháng 30 ngày nhưng policy không nêu quy ước đó. Output vẫn thiếu số tiền theo tháng có thể tính được.
- **Fix:** Tính rõ 15 triệu × 2% = 300.000 VNĐ/tháng; nói rõ thiếu quy ước chia ngày. Lưu NLI verdict, kiểm tra dữ kiện query/phép tính và xác minh pro-rata với chủ policy trước khi dùng.

### #5 — Câu 7: Chu kỳ đổi mật khẩu

- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Hiện hành mỗi 120 ngày; ground truth còn nhắc bản cũ 90 ngày.
- **Got:** Trả lời đúng mỗi 120 ngày [2].
- **Contexts:** `so_tay_an_toan.pdf` → `mat_khau_v2.md` → `chi_phi_expense.md`.
- **Worst metric:** context_precision = 0,5000.
- **Error Tree:** Output đúng hiện hành? Có → Nguồn đầu liên quan? Không → Nguồn thứ hai có 120 ngày? Có → Context có 90 ngày bản cũ? Không → Ranking và phạm vi đánh giá phiên bản.
- **Diagnosis:** Chính sách mật khẩu hiện hành đứng trước theo thứ tự nguồn khác nhau ở RRF, nhưng reranking đưa sổ tay an toàn lên đầu. Generator đọc nguồn thứ hai nên answer đúng; precision thấp vì noise. Bản v2 chỉ nói thay thế v1, không ghi giá trị 90 ngày; phần lịch sử trong ground truth thiếu ở context cuối.
- **Fix:** Đo reranker đa ngôn ngữ cho truy vấn quy định thời gian, MRR/Coverage@3. Query lịch sử lấy hai phiên bản và nêu rõ bản bị thay thế; query hiện hành đánh giá riêng chính sách hiện hành thay vì trộn yêu cầu so sánh lịch sử.

## Case Study: Câu 4

1. Output thiếu 15 ngày, dù Faithfulness = 1.
2. Ba context cuối thiếu chính sách phép năm hiện hành.
3. Query rõ; nguồn đúng ở candidates nhưng parent đứng thứ 5 sau ranking.
4. Ưu tiên M3/chọn context; đo candidate recall và coverage sau ranking riêng.

## Một giờ tiếp theo

- Lưu NLI/context usefulness verdict, kiểm tra dữ kiện query, phép tính và ground truth nhiều ý; không dùng điểm thấp làm chẩn đoán chắc chắn khi thiếu verdict.
- Đo BGE/reranker đa ngôn ngữ trên holdout khi tải xong, ưu tiên coverage nhiều nguồn và phiên bản; không tối ưu riêng năm câu đã thấy.
- Kiểm tra quota ngày trước full run; pacing chỉ xử lý quota phút. Dùng `main.py --eval-only` cho ô thiếu, giữ điểm/inputs cũ nguyên vẹn.
- Baseline generation timing thuộc model cũ; evaluator timing thuộc lần chấm mới. Chênh lệch không chỉ do pipeline vì generation model khác nhau.
