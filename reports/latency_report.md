# Latency breakdown - Lab 18

Run recorded: 2026-10-04T21:53:54.259393+07:00.

Profile: lightweight retrieval models with Gemini generation, enrichment and RAGAS.
Embedding: sentence-transformers/all-MiniLM-L6-v2.
Reranker: cross-encoder/ms-marco-TinyBERT-L2-v2.
LLM: gemini-3.1-flash-lite.
RAGAS embeddings: gemini-embedding-001.
Baseline generation/build timings come from the saved original run; evaluator timings come from the current run.
Production reused validated enrichment and saved matching checkpoints when available.
Baseline generation model: gemini-3.5-flash-lite.
Shared RAGAS evaluator: gemini-3.1-flash-lite.
Enrichment model: gemini-3.5-flash-lite.
Index timings include encoder loading as needed; these runs do not share an encoder process.
Generation includes API/network latency and quota waits.
Production reused all 101 previously generated Gemini enrichment payloads.
The enrichment build timing below measures cache validation, not the original API calls.
RAGAS evaluation includes all four metrics over 20 questions and recorded retries.

| Build step | Baseline (ms) | Production (ms) |
|---|---:|---:|
| load_and_chunk_ms | 107.12 | 115.85 |
| enrichment_ms | N/A | 0.44 |
| index_ms | 22529.18 | 17849.15 |
| reranker_load_ms | N/A | 3266.30 |

| Pipeline | Query step | Samples | Mean (ms) | p50 (ms) | p95 (ms) | Max (ms) |
|---|---|---:|---:|---:|---:|---:|
| Baseline | retrieval_ms | 20 | 48.64 | 46.13 | 66.35 | 195.35 |
| Baseline | generation_ms | 20 | 4138.21 | 4223.95 | 4790.45 | 7563.12 |
| Production | retrieval_ms | 20 | 48.20 | 44.12 | 66.44 | 67.13 |
| Production | rerank_ms | 20 | 18.05 | 17.42 | 22.74 | 29.63 |
| Production | generation_ms | 20 | 4971.12 | 4309.97 | 9686.67 | 13980.71 |

| Evaluation | Time (seconds) |
|---|---:|
| Baseline RAGAS | 906.68 |
| Baseline additional metric retries | 41.57 |
| Production RAGAS | 848.60 |

Generation backends: {"Baseline": {"gemini": 20}, "Production": {"gemini": 20}}.
Enrichment backends: {"gemini": 101}.
Raw timings and source traces are preserved in the two JSON reports.
