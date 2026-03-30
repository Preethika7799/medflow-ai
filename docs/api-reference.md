# HTTP API Reference

Base URL: `http://localhost:8000`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness (`{"status": "healthy"}`) |
| GET | `/ready` | Readiness (+ Qdrant ping) |
| POST | `/api/v1/documents/upload` | Multipart upload → ingestion |
| GET | `/api/v1/documents` | List documents (aggregated from Qdrant) |
| GET | `/api/v1/documents/{id}` | Document chunk previews |
| POST | `/api/v1/query` | RAG query (`query`, optional `filters`, `strategy`) |
| POST | `/api/v1/evaluate` | Run golden evaluation (long-running) |

Open **/docs** for interactive OpenAPI + schemas with examples.
