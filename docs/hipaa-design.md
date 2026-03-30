# PHI handling in this codebase

The repo only ships **synthetic** patient-like strings. Nothing here is a HIPAA certification or legal review.

**In the pipeline**

1. **De-ID before LLM calls** — `DeIDPipeline` runs Presidio (plus custom recognizers) before classification, chunking, and answer generation.
2. **Logging** — structured logs record coarse events (ingestion, routing). Raw clinical blobs are not written to logs by default.
3. **Deploying for real data** — you would terminate TLS at the edge, lock down network paths, and use vendor agreements and key management appropriate to your org. That work lives outside this repo.

**Out of scope here**

- BAAs, full IAM/KMS design, retention policy, break-glass re-identification process.
