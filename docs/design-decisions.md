# Design Decisions

## 1. Qdrant (self-hosted) over managed Pinecone/Weaviate

**Why:** predictable cost, Docker-friendly local dev, strong metadata filtering, and cosine/dot distance configuration without vendor lock-in.

**Tradeoff:** you operate backups and capacity; managed SaaS would reduce ops at higher recurring cost.

## 2. PaddleOCR primary with EasyOCR fallback

**Why:** Paddle offers competitive accuracy on scanned forms; EasyOCR provides a pragmatic second opinion when confidence is low.

**Tradeoff:** two OCR stacks add weight; GPU helps throughput if you batch heavily.

## 3. Hybrid retrieval (dense + BM25 + RRF)

**Why:** clinical queries mix semantic paraphrase (“low back MRI”) with exact tokens (CPT codes, medication strings).

**Tradeoff:** requires maintaining a BM25 snapshot synced from Qdrant; extra CPU for fusion and re-ranking.

## 4. Presidio before any LLM boundary

**Why:** minimizes accidental PHI egress to third-party LLM APIs; aligns with common HIPAA-oriented engineering controls for demos.

**Tradeoff:** Presidio + spaCy models add install complexity; heuristic recognizers can miss novel patterns.

## 5. Provider abstraction with OpenAI, Anthropic, and Ollama

**Why:** swap backends without touching call sites—Ollama for local runs, hosted APIs when you care about quality.

**Tradeoff:** each backend has subtly different token accounting and capability gaps (tool calling, streaming).

## 6. Langfuse (optional) + structlog

**Why:** Langfuse tracks traces for tuning prompts; structlog gives JSON-friendly server logs with correlation IDs.

**Tradeoff:** self-hosted Langfuse requires Postgres; adds stack surface area.
