# Architecture

## Diagram

```mermaid
flowchart TB
  subgraph intake [Ingestion]
    U[Upload API] --> L[Loaders PDF/Image/Text]
    L --> O[OCR Pipeline Paddle/EasyOCR]
    O --> D[De-ID Presidio + custom recognizers]
    D --> C[LLM Classifier]
    C --> K[Chunkers fixed/recursive/semantic]
    K --> E[Embeddings sentence-transformers]
    E --> Q[(Qdrant)]
  end

  subgraph rag [Query path]
    QU[POST /query] --> RT[Query router LLM]
    RT --> DN[Dense retriever]
    RT --> SP[BM25 retriever]
    RT --> HY[RRF hybrid]
    DN --> RR[Cross-encoder rerank]
    SP --> RR
    HY --> RR
    RR --> GEN[Response generator]
    GEN --> OUT[Answer + citations]
    Q -.-> DN
    Q -.-> SP
  end

  subgraph obs [Observability]
    LF[Langfuse traces]
    AUD[Audit + structlog]
    MET[Token/cost metrics]
  end

  GEN --- LF
  D --- AUD
  GEN --- MET
```

## Components

- **API (`src/api`)** — FastAPI application with ingestion, RAG query, evaluation trigger, health checks.
- **Core (`src/medflow`)** — OCR, de-identification, retrieval, generation, evaluation, and provider abstractions.
- **Dashboard (`src/dashboard`)** — Streamlit operational views for documents, QA, metrics, and health.
- **Data (`data/synthetic`)** — Synthetic corpus + golden Q&A for reproducible evaluation.
