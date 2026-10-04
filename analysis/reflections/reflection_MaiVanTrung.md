# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Mai Văn Trung
**MSSV:** 2A202602513
**Khóa:** K4 - Track 3A
**Ngày thực hiện:** 04/10/2026 (GMT+7)

Bản reflection này ghi lại implementation và quan sát từ lần kiểm tra repository.
Action plan dưới đây là đề xuất cho project tra cứu chính sách nội bộ dựa trên corpus
của lab; không giả định có một hệ thống cá nhân khác đang vận hành.

**Cấu hình chạy thực tế:** model nhẹ MiniLM-L6-v2 và TinyBERT-L2-v2.
Gemini 3.5 Flash-Lite hết quota ngày; sau lựa chọn người dùng, `.env` đổi sang
`gemini-3.1-flash-lite`. Đã có xác nhận cho phép gửi dữ liệu bài lab tới Google.
RAGAS chấm baseline/production cùng Gemini 3.1, embedding `gemini-embedding-001`,
đủ 20/20 điểm hữu hạn cho mỗi metric; không dùng synthetic probe làm điểm lab.

Production: Faithfulness 0,9317; Relevancy 0,9009; Precision 0,6833; Recall 0,8000.
Baseline chấm lại: 0,9679; 0,5075; 0,3583; 0,4333. Precision chưa đạt 0,70.
Baseline giữ câu trả lời Gemini 3.5, production tạo mới bằng Gemini 3.1;
chênh lệch không chỉ do retrieval/reranking. Không làm tròn/chọn lại điểm.

Script chuyển evaluator và `main.py --eval-only` exit code 0. Pipeline source
đã chạy corpus thật qua script; standalone trước đó được kiểm tra offline trong
thư mục riêng với Qdrant bộ nhớ. Bộ test có 73 trường hợp; kết quả kiểm tra cuối
ở `reports/validation_report.json`. Reranking trung bình 18,05 ms/query,
generation khoảng 4971 ms/query, gồm network và chờ quota. Dùng lại 101 payload
enrichment Gemini 3.5 sau khi kiểm tra source/parent/version; timing enrichment
là xác minh cache, không phải 101 API calls ban đầu.

### Lỗi quota ngày và khôi phục đánh giá

Probe chat trả HTTP 429 với quota `GenerateRequestsPerDayPerProjectPerModel-FreeTier`,
limit 500 và thời gian chờ khoảng 9 giờ 52 phút; embedding vẫn trả vector 3072 chiều.
Giảm RPM chỉ xử lý quota phút. Sau lựa chọn người dùng, Gemini 3.1 probe thành
công; chấm lại toàn bộ bằng evaluator mới, giữ riêng lần lỗi và điểm cũ cùng
đúng inputs. Hai ô baseline câu 16 lỗi kết nối được retry riêng; production
chấm đủ không cần retry metric.

RAGAS trước đây log lỗi rồi trả NaN. M4 nay lưu lỗi đã redact key trong JSON,
dừng checkpoint/retry khi hết quota ngày, phân biệt `unavailable` và `partial`.
`main.py` probe chat/embedding trước full run mới, giữ báo cáo nếu probe thất
bại; `--eval-only` chỉ chấm ô thiếu, không gọi API nếu đủ điểm. `save_report()`
lưu bản hoàn chỉnh gần nhất trước khi ghi kết quả mới. Tests kiểm tra giữ điểm,
answer/context, không lộ key, không tiêu quota thật.

## Phần 1: Mapping bài giảng

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()`, `compare_strategies()` | Với threshold 0,85, corpus gộp cho 208 semantic chunks, so với 51 basic chunks. Threshold này chia khá nhỏ; mô hình MiniLM không bảo đảm các câu tiếng Việt cùng chủ đề đạt similarity cao. Cần thử threshold thấp hơn rồi đo retrieval thay vì giả định semantic luôn tạo ít chunks. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Pipeline chia riêng từng tài liệu: 26 parents và 101 children, children tối đa 256 ký tự. ID gồm hash của source + text để tránh trùng giữa tài liệu. `run_query()` trả lại parent và loại parent trùng, giữ điều kiện/phủ định nằm ngoài child. |
| BM25 + Dense fusion | M2 | `segment_vietnamese()`, `DenseSearch.index/search()`, `reciprocal_rank_fusion()` | Cùng chuẩn hóa NFC, chữ thường, dấu câu cho corpus và query; thay underscore của underthesea. RRF cộng 1/(60 + rank + 1), không cộng trực tiếp hai thang điểm BM25/cosine. Test kiểm tra merge, score, tài liệu trùng text khác nguồn và dense index/query trên Qdrant thật trong bộ nhớ. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker._load_model/rerank()`, `benchmark_reranker()` | CrossEncoder đọc cặp query/document. Retrieval tìm bằng enriched text, nhưng reranking dùng original child; parent expansion diễn ra sau xếp hạng. Cache model dùng lại giữa các instance để không tính chi phí tải lại trong mỗi query. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()`, `retry_report_missing_metrics()` | Bốn metrics đều có 20/20 mẫu. Câu 4 có faithfulness 1 nhưng relevancy/precision/recall 0: answer bám context sai chủ đề vẫn không giải quyết query. Bottom-5 thật là câu 4, 13, 5, 12, 7. Chỉ retry ô thiếu; tests kiểm tra điểm hợp lệ không bị thay bởi lần retry. |
| Contextual embeddings | M5 | `contextual_prepend()`, `_enrich_single_call()`, `enrich_chunks()` | Combined mode lấy summary, questions, context, metadata qua một call/chunk. Cả 101 payload có backend Gemini và được khôi phục sau interruption. Enriched text dùng để tìm, original child dùng để rerank, parent dùng để trả lời; lineage nguồn được bảo vệ. Chưa có ablation nên không quy mọi cải thiện điểm riêng cho enrichment. |

Nguồn số liệu chunking: `reports/chunking_comparison.json`.
So sánh này gộp corpus chỉ để khảo sát; production luôn chia theo từng tài liệu,
nên số parent/child của production khác bảng comparison.

## Phần 2: Khó khăn & cách giải quyết

### Docker chưa chạy

Thông báo ban đầu:
`failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine`,
kèm `The system cannot find the file specified.`

Kiểm tra bằng `docker info` xác nhận Docker daemon chưa sẵn sàng.
Khởi động Docker Desktop, chạy `docker compose up -d`, rồi kiểm tra
`http://127.0.0.1:6333/healthz`: nhận `healthz check passed`.
Code vẫn hỗ trợ Qdrant trong bộ nhớ nếu server không sẵn sàng.

### Pytest không ghi được thư mục tạm

Lỗi thực tế:
`PermissionError: [WinError 5] Access is denied: 'C:\Users\Admin\AppData\Local\Temp\pytest-of-Admin'`.

29 tests đã pass nhưng một test report lỗi ở fixture `tmp_path`, trước khi gọi
implementation. Đọc traceback xác định lỗi quyền filesystem, không phải lỗi JSON.
Đặt `--basetemp=.cache/pytest` trong `pyproject.toml`; chạy lại nhóm kiểm tra
đã pass. Các file tạm giờ nằm trong workspace.

### Tải model và kiểm tra đúng artifact

Hugging Face báo `The read operation timed out` khi tải tokenizer/trọng số.
Kiểm tra danh sách file model chính thức cho thấy BGE-M3 có
`pytorch_model.bin`, còn reranker có `model.safetensors`.
Chỉ lọc safetensors sẽ thiếu trọng số BGE-M3 dù các file cấu hình đã tải thành công.

Bổ sung file bin cần thiết, loại ONNX/OpenVINO khỏi download mặc định, dùng cache
trong dự án. Khi transport mặc định bị treo, script hỗ trợ
`--transport ranged`: tải từng phần, kiểm tra Content-Range và SHA256 trước
khi lắp ghép trọng số. MiniLM được dùng lại từ cache có sẵn để kiểm tra M1.
Đường truyền chậm với khoảng 4,3 GB trọng số BGE. Sau lựa chọn của học viên,
đã dừng tải BGE, giữ partial cache và chạy trước profile nhẹ. Script cũng có
`--transport stream` để tiếp tục tải tuần tự có resume và kiểm tra SHA256.

### Thiếu API key và nguy cơ báo cáo sai

Thông báo của evaluator: `OPENAI_API_KEY is not configured`.

Fallback extractive cho phép kiểm tra pipeline nhưng không thay thế LLM trả lời
hoặc RAGAS judge. Giữ đủ question/answer/contexts/ground_truth, đánh dấu unavailable,
serialize metric chưa đo thành null và in N/A khi so sánh. Diagnostic Tree chỉ
xếp bottom-N khi có đủ bốn metric hữu hạn; không gán diagnosis từ điểm giả.

Sau đó đã bổ sung hỗ trợ Gemini. RAGAS cần nhiều completions nhưng Gemini không
nhận `n > 1`, nên wrapper gửi từng completion riêng và giữ cấu trúc generations.
Probe nhân tạo xác nhận API hoạt động, nhưng không được dùng điểm đó cho bài lab.
Theo lựa chọn chỉ xử lý trên máy, cấu hình chuyển sang `offline`; tests kiểm tra
không tạo API client dù `.env` vẫn có cloud key. Provider `local` hỗ trợ endpoint
OpenAI trên loopback và embeddings chạy bằng SentenceTransformer trên máy;
endpoint cloud hoặc địa chỉ LAN bị từ chối để tránh gửi corpus ra ngoài máy.
Sau đó người dùng đã xác nhận rõ cho phép gửi dữ liệu bài lab tới Gemini và
cấu hình chuyển sang `LLM_PROVIDER=gemini` để chạy đánh giá corpus thật.

### Gemini vượt quota requests/phút

Lần chạy corpus đầu nhận lỗi:
`RateLimitError: Error code: 429`, `RESOURCE_EXHAUSTED`,
`GenerateRequestsPerMinutePerProjectPerModel-FreeTier`, `quotaValue: 15`.

Đọc chi tiết lỗi xác định giới hạn là chat requests/phút, không phải lỗi API key.
RAGAS có thể phát nhiều request trong một metric, nên chỉ giảm số workers không
đảm bảo tuân thủ quota. Dừng lần chạy đó, bổ sung `RequestPacer` dùng chung cho
HTTP hooks đồng bộ/bất đồng bộ của enrichment, generation và RAGAS. Mặc định
14 request/phút, chừa biên dưới quota 15; retries của SDK cũng đi qua hook.
Tăng timeout mỗi metric lên 300 giây để thời gian chờ quota không gây timeout giả.
Tests dùng đồng hồ giả và MockTransport kiểm tra slot dùng chung và HTTP retry,
không gửi request thật hoặc chờ theo thời gian thực.

### Lần chạy bị ngắt và metric thiếu

Baseline có `APIConnectionError(Connection error.)` và `TimeoutError()`, thiếu
faithfulness ở câu 2/19 và precision ở câu 10. Production gặp
`ValidationError: ContextPrecisionVerifications`, `Failed to parse output. Returning None.`
ở precision câu 13. Không biến những ô đó thành 0 hoặc coi report đã hoàn tất.

Khôi phục 101 enrichment payloads từ volume Qdrant, đối chiếu từng child với
source/parent/version trước khi dùng. Thêm checkpoint câu trả lời và điểm theo
từng câu; fingerprint bao gồm dữ liệu, model và prompt. Chấm bổ sung đúng các ô
thiếu trên answer/context giữ nguyên; mọi điểm đã hữu hạn được giữ lại. Tests
kiểm tra resume không gọi lại generation và dữ liệu đổi thì không dùng checkpoint
cũ. Lần hoàn tất cả hai reports có status `complete`, mỗi metric đủ 20 mẫu.

### Kiến thức cần bổ sung

Cần hiểu rõ sự khác nhau giữa lỗi retrieval và lỗi generation: một câu trả lời
trích nguyên parent vẫn có thể không trả lời đủ câu hỏi nhiều bước.
Cần phân biệt chi phí model load với inference, hiểu RRF dựa trên rank, kiểm tra
tính tương thích của RAGAS với Dataset thực tế và kiểm tra lineage của metadata.
Cách bổ sung là đọc API/source đã cài, viết test cho các điểm giao tiếp và đo trên
tập câu hỏi độc lập. Cần lưu NLI verdict từng statement để phân biệt lỗi judge với
lỗi answer: dữ kiện số từ query và phạm vi so sánh chính sách cũ/mới có thể ảnh
hưởng metrics. Không kết luận hallucination chỉ từ một score thấp.

## Phần 3: Action plan cho project

### Project đề xuất: Trợ lý tra cứu chính sách nội bộ tiếng Việt

### Hiện trạng

Pipeline đã có hierarchical chunking → combined enrichment → BM25 + dense → RRF
→ CrossEncoder → parent context → grounded answer → evaluation.
Corpus có chính sách cũ/mới, điều kiện phủ định, ngưỡng phê duyệt, câu hỏi nhiều
nguồn và phép tính. Hai PDF scan chưa có text nên chưa được tìm kiếm.
Gemini đã tổng hợp được nhiều nguồn, nhưng bảng lương của câu 12 bị loại sau
reranking dù có trong candidates. Câu 4 mất nguồn phép năm đúng sau ranking.
Các câu phiên bản còn có khác biệt giữa scope hiện hành của query và phần so
sánh lịch sử trong ground truth. Đây là các known issues của profile nhẹ.

### Kế hoạch áp dụng

1. **Chunking:** dùng section của Markdown làm ranh giới trước khi chia hierarchy,
   giữ bảng và phủ định cùng điều kiện; thử semantic threshold 0,5/0,65/0,85 trên
   tập validation. Thêm OCR riêng cho PDF scan, lưu số trang và đánh giá chất lượng OCR.
2. **Search:** dùng BM25 + BGE-M3 + RRF. Đo Recall@20 trên tập có nhãn nguồn.
   Bổ sung effective_from/effective_to để hỗ trợ câu hỏi lịch sử và phiên bản tương lai.
3. **Reranking:** dùng bge-reranker-v2-m3; đo p50/p95 trên phần cứng mục tiêu,
   so với Flashrank nếu độ trễ không đạt yêu cầu. Giữ các nguồn khác nhau cho multi-hop.
4. **Evaluation:** dùng lần đo Gemini hoàn tất làm baseline cho thử nghiệm tiếp,
   giữ cùng answer generator và quota pacing; kiểm tra bottom-5 và nhóm version/negation/numeric.
   Tách scope hiện hành/lịch sử, kiểm tra quy ước pro-rata câu 17 trước khi gắn nhãn số tiền.
   Thêm tập holdout và kiểm tra thủ công phép tính, không tối ưu chỉ trên 20 câu lab.
5. **Enrichment:** dùng combined contextual + summary + HyQA cho tài liệu khó tìm.
   Cache theo hash nội dung + model/prompt version, đo ablation có/không enrichment
   và chi phí trước khi bật toàn corpus.

### Timeline

- **Tuần 1 (05–11/10/2026):** gắn nhãn nguồn cho validation/holdout và kiểm tra NLI verdict,
  bổ sung section hierarchy và OCR hai PDF scan.
- **Tuần 2 (12–18/10/2026):** thử threshold và ablation retrieval/enrichment,
  so sánh reranker về Recall@k, precision, latency và chi phí.
- **Tuần 3 (19–25/10/2026):** bổ sung cache, kiểm thử câu hỏi lịch sử và nhiều nguồn,
  citation theo trang/section; chọn cấu hình dựa trên holdout.
