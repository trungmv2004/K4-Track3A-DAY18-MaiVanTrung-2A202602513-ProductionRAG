# Lab 18: Production RAG Pipeline

**K4-Track3A · Ngày 18 · Production RAG**  
**Thời gian:** 2h implement + 30 phút reflection

---

## Tổng quan

Bài tập **cá nhân** — implement toàn bộ 5 modules:

```
M1 Chunking → M5 Enrichment → M2 Hybrid Search → M3 Reranking → LLM Answer → M4 RAGAS Eval
```

Xem **ASSIGNMENT.md** để biết chi tiết từng module và timeline.

## Prerequisites

| Dependency | Bắt buộc? | Dùng cho |
|-----------|-----------|----------|
| Docker (Qdrant) | ✅ Có | M2 Dense Search |
| Python 3.11+ | ✅ Có | Tất cả modules (RAGAS cần 3.11+ cho asyncio) |
| `OPENAI_API_KEY` hoặc `GEMINI_API_KEY` | ⚠️ M4+M5 | Trả lời, RAGAS eval (M4), Enrichment LLM (M5) |

**Pre-download models** (tránh timeout trong lab):
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"
```

## Quick Start

### 1. Clone repository & tạo môi trường ảo

**Linux / macOS / Git Bash:**
```bash
git clone <repo-url>
cd K4-Track3A-Production-RAG
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
git clone <repo-url>
cd K4-Track3A-Production-RAG
python -m venv .venv
.venv\Scripts\Activate.ps1
```
*(Nếu dùng Windows CMD: chạy `.venv\Scripts\activate.bat`)*

### 2. Cài đặt dependencies & Khởi động dịch vụ

**Linux / macOS / Git Bash:**
```bash
docker compose up -d                    # Khởi động Qdrant vector database
pip install -r requirements.txt
cp .env.example .env                    # Tạo file .env và điền OPENAI_API_KEY
python naive_baseline.py                # Khởi tạo baseline
```

**Windows (PowerShell):**
```powershell
docker compose up -d                    # Khởi động Qdrant vector database
pip install -r requirements.txt
Copy-Item .env.example .env             # Tạo file .env và điền OPENAI_API_KEY
python naive_baseline.py                # Khởi tạo baseline
```
*(Nếu dùng Windows CMD: dùng `copy .env.example .env` thay cho `Copy-Item`)*

## Chạy toàn bộ & Kiểm tra

```bash
python main.py                          # Chạy Naive + Production + In bảng so sánh
python check_lab.py                     # Script kiểm tra hợp lệ trước khi nộp (chạy được trên mọi OS)
```

Khi lần chạy bị ngắt sau khi đã có báo cáo baseline/enrichment, có thể chạy
`python main.py --resume`. Chế độ này giữ điểm RAGAS đã đo, chỉ chấm các ô thiếu;
production lưu checkpoint sau từng câu trả lời và từng câu được đánh giá trong
`.cache/`. Cache enrichment phải khớp nội dung/parent/metadata nguồn hiện tại;
checkpoint chỉ được dùng lại khi fingerprint dữ liệu, model và prompt khớp.

Nếu câu trả lời đã có và chỉ cần chấm bổ sung metric, dùng
`python main.py --eval-only`: kiểm tra kết nối chat/embedding rồi chỉ chạy các ô
thiếu trong hai báo cáo, không tạo lại enrichment hoặc câu trả lời. Provider,
model LLM và model embedding phải trùng với báo cáo để tránh trộn điểm evaluator.

Gemini có cả quota phút và quota ngày. Giảm `GEMINI_REQUESTS_PER_MINUTE` chỉ
giải quyết quota phút; lỗi `GenerateRequestsPerDay` cần chờ quota được cấp lại
hoặc chọn model khác còn quota. `main.py` kiểm tra provider trước lần chạy mới
để tránh ghi đè báo cáo bằng kết quả thiếu khi API không khả dụng. Lỗi RAGAS
được lưu trong `evaluation_error`/`evaluation_errors`, cùng số mẫu hợp lệ mỗi
metric; lượt chạy chưa đủ điểm trả exit code 1.
Trước khi ghi báo cáo mới, bản hoàn chỉnh đang có được lưu thành
`reports/last_complete_<tên báo cáo>.json`, kèm đúng câu trả lời và context đã chấm.

Để chuyển evaluator Gemini sau khi model cũ hết quota, đặt `LLM_MODEL` trong
`.env` rồi chạy `python scripts/rerun_gemini_evaluation.py`. Script chấm lại toàn
bộ baseline đã lưu và production bằng cùng evaluator, tạo lại câu trả lời
production, đồng thời tái sử dụng `.cache/gemini_enriched_index.json` đã kiểm tra
nội dung. Nó ghi riêng model tạo enrichment và model tạo câu trả lời baseline.
Không ghép điểm evaluator cũ vào bộ điểm của model mới.

### Bản triển khai trong repository này

- M1–M5 đã có implementation. Pipeline tìm child, khôi phục parent, loại parent trùng
  và ưu tiên chính sách hiện hành theo phiên bản/ngày hiệu lực của tài liệu.
- Dùng Python trong môi trường ảo để tránh cài nhầm dependencies:
  `.venv\Scripts\python.exe main.py` và `.venv\Scripts\python.exe check_lab.py`.
- Tải trước các model bằng `.venv\Scripts\python.exe scripts/download_models.py`.
  Cache nằm ở `.cache/models/`, không đưa vào Git.
  Nếu transport mặc định bị timeout, thêm `--transport ranged` để tải từng phần
  có resume và kiểm tra SHA256. Có thể cài lại đúng môi trường đã kiểm thử bằng
  `pip install -r requirements-lock.txt`.
- Cấu hình `OPENAI_API_KEY` hoặc `GEMINI_API_KEY` trong `.env`. Nếu cả hai để trống, enrichment dùng
  fallback extractive, câu trả lời là trích đoạn nguồn, và RAGAS có trạng thái
  `unavailable`; metric trong JSON là `null` và bảng so sánh là `N/A`.
  Đây không phải điểm đánh giá thật. Điền key rồi chạy lại `main.py` để đo RAGAS.
- Chỉ có Gemini key thì provider tự chọn Gemini. Có thể đặt `LLM_PROVIDER=gemini`,
  `GEMINI_MODEL=gemini-3.5-flash-lite`; evaluator dùng `gemini-embedding-001`.
  `LLM_MODEL` để trống để dùng mặc định của provider, hoặc đặt model phù hợp.
  Tích hợp qua [endpoint tương thích OpenAI chính thức của Google](https://ai.google.dev/gemini-api/docs/openai),
  nên không cần cài SDK mới. RAGAS gửi completion riêng khi cần nhiều câu hỏi giả thuyết.
  Tests luôn dùng key giả/mocks và không tiêu quota từ `.env`.
- Lần chạy thực tế với Gemini trả quota 15 chat requests/phút. Cấu hình
  `GEMINI_REQUESTS_PER_MINUTE=14` mặc định điều tiết chung enrichment, generation
  và RAGAS, kể cả các lời gọi đồng bộ/bất đồng bộ và retry HTTP. Có thể điều chỉnh
  theo quota của tài khoản; đánh giá đủ 20 câu hỏi cho hai pipeline mất thời gian
  vì mỗi metric cần nhiều lời gọi. Điểm thiếu vẫn được ghi `null`, không thay bằng 0.
- Chỉ xử lý trên máy: đặt `LLM_PROVIDER=offline` để dùng extractive fallback,
  không gọi API LLM/embedding trên cloud dù `.env` có key. RAGAS vẫn là `unavailable`
  vì embedding/reranker không thay thế được model sinh văn bản dùng chấm điểm.
- Khi đã có server LLM local tương thích OpenAI (ví dụ Ollama), đặt
  `LLM_PROVIDER=local`, `LOCAL_BASE_URL=http://127.0.0.1:11434/v1` và
  `LOCAL_MODEL=<model đã cài trên server>`, để `LLM_MODEL` trống.
  Endpoint local bắt buộc là loopback; embeddings đánh giá dùng SentenceTransformer
  trên máy (`EVAL_EMBEDDING_MODEL` hoặc `EMBEDDING_MODEL`), không dùng Gemini key.
  Tải trước model và đặt `HF_HUB_OFFLINE=1` khi chạy để tránh tải thêm từ Hugging Face.
  Kết quả RAGAS còn phụ thuộc khả năng model local trả đúng JSON của evaluator.
- Nếu Qdrant server chưa sẵn sàng, dense search dùng Qdrant trong bộ nhớ và in
  thông báo. Dùng `docker compose up -d` để chạy Qdrant server theo đề bài.
- Báo cáo lưu đủ 20 câu hỏi, câu trả lời, contexts, nguồn, cấu hình và latency.
  PDF scan vẫn cần OCR trước khi có thể được tìm kiếm.

**Cấu hình lần chạy hiện tại (model nhẹ, theo lựa chọn của học viên):**

```dotenv
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANK_MODEL=cross-encoder/ms-marco-TinyBERT-L2-v2
```

Hai giá trị này đã được cấu hình trong `.env` local. Đây là cấu hình kiểm tra
pipeline; chất lượng tiếng Việt cần đánh giá riêng và không đại diện cho BGE-M3.
Để chạy model chuẩn, đặt lại hai biến theo `.env.example`, tải model bằng
`scripts/download_models.py --transport stream`, rồi chạy lại `main.py`.
Không commit `.env`, cache hoặc API key.

## Cấu trúc repo

```
K4-Track3A-Production-RAG/
├── README.md                   # File này
├── ASSIGNMENT.md               # ★ Đề bài + timeline + reflection
├── RUBRIC.md                   # Hệ thống chấm điểm
│
├── main.py                     # Entry point: chạy toàn bộ pipeline
├── check_lab.py                # Kiểm tra định dạng trước khi nộp
├── naive_baseline.py           # Baseline (chạy trước)
├── config.py                   # Shared config
├── requirements.txt            # Dependencies
├── docker-compose.yml          # Qdrant local
├── .env.example                # API keys template
│
├── data/                       # Corpus tiếng Việt — 25 .md files + 3 PDFs (28 files total)
│   ├── nghi_phep_nam_v2023.md  # Nghỉ phép 12 ngày (v2023, superseded)
│   ├── nghi_phep_nam_v2024.md  # Nghỉ phép 15 ngày (v2024, hiện hành)
│   ├── mat_khau_v1.md          # Password policy 90 ngày (OLD)
│   ├── mat_khau_v2.md          # Password policy 120 ngày + MFA (NEW)
│   ├── ... (28 files total)    # 8 categories: leave, salary, IT, workflow, training, admin, safety, compliance
│   ├── so_tay_an_toan.pdf      # An toàn PCCC + sơ cứu (PDF text)
│   ├── BCTC.pdf                # Báo cáo tài chính (scan, cần OCR)
│   └── Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf # Nghị định BVDL (scan, cần OCR)
├── test_set.json               # 20 Q&A pairs (6 types: lookup, version, negation, multi-hop, numeric, ambiguous)
│
├── src/                        # ★ Scaffold code (có TODO markers)
│   ├── m1_chunking.py          # Module 1: Chunking
│   ├── m2_search.py            # Module 2: Hybrid Search
│   ├── m3_rerank.py            # Module 3: Reranking
│   ├── m4_eval.py              # Module 4: Evaluation
│   ├── m5_enrichment.py        # Module 5: Enrichment Pipeline
│   └── pipeline.py             # Ghép toàn bộ pipeline
│
├── tests/                      # Auto-grading
│   ├── test_m1.py
│   ├── test_m2.py
│   ├── test_m3.py
│   ├── test_m4.py
│   └── test_m5.py
│
├── analysis/                   # ★ Deliverable
│   ├── failure_analysis.md     # Phân tích failures (cá nhân)
│   └── reflections/            # Reflection cá nhân
│       └── reflection_TEMPLATE.md
│
├── reports/                    # ★ Auto-generated (bắt buộc: reports/ragas_report.json)
│   ├── ragas_report.json
│   └── naive_baseline_report.json
│
└── templates/                  # Templates gốc (backup)
    └── failure_analysis.md
```

## Timeline (Thời lượng ước tính)

| Thời lượng | Hoạt động |
|------------|-----------|
| 10 phút | Setup môi trường + chạy `naive_baseline.py` |
| 90 phút | Implement M1 → M2 → M3 → M4 → M5 |
| 20 phút | Chạy pipeline + RAGAS + failure analysis |
| 30 phút | Reflection: lecture mapping + project plan |

## Quy chuẩn đặt tên Repository & Nộp bài

- **Cấu trúc đặt tên repo:**  
  `K4-Track3A-DAY18-<HoVaTen>-<MSSV>-ProductionRAG`  
  *(Ví dụ: `K4-Track3A-DAY18-NguyenVanAn-AI20K001-ProductionRAG`)*
- **Hạn chót nộp bài:** **23h59 ngày diễn ra bài lab (GMT+7)** trên cổng VLearn LMS / Codelab.
- **Chi tiết yêu cầu:** Xem tại [ASSIGNMENT.md](ASSIGNMENT.md) và [RUBRIC.md](RUBRIC.md).
