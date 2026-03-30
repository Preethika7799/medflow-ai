# Manual testing playbook

Rough order matters. Full pass is ~1–2 hours if nothing breaks.

- Package name on disk is `medflow-ai`; imports are `medflow` and `api`. After `pip install -e ".[dev,dashboard]"`, `pip show medflow-ai` should find it.
- Synthetic PDFs are named like `doc_prior_auth_001.pdf`.
- `GET /health` → `{"status": "healthy"}`.
- `make run` = `uvicorn api.main:app` from repo root with venv on.

---

## PHASE 0: Environment Setup

### Step 0.1 — Python 3.11

```powershell
python --version
py -3.11 --version
```

Install 3.11 via [python.org](https://www.python.org/downloads/release/python-3119/), **pyenv-win**, or the **py** launcher.

### Step 0.2 — Create virtual environment

```powershell
cd Medflow_ai

py -3.11 -m venv .venv
.venv\Scripts\activate
python --version
```

### Step 0.3 — Install dependencies

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev,dashboard]"
```

**This takes a while** (PaddleOCR, PyTorch, sentence-transformers).

**If you hit errors:**

```powershell
pip install paddlepaddle
pip install paddleocr

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers

pip install spacy
python -m spacy download en_core_web_sm
pip install presidio-analyzer presidio-anonymizer

pip install -e ".[dev,dashboard]"
```

**CHECKPOINT:** `pip show medflow-ai` shows the project installed.

### Step 0.4 — Environment variables

```powershell
copy .env.example .env
```

Configure at least one LLM path:

```env
OPENAI_API_KEY=sk-...
# OR
ANTHROPIC_API_KEY=sk-ant-...
# OR use Ollama (MEDFLOW_ENV=development in configs/development.yaml uses ollama) + run Ollama locally

QDRANT_HOST=localhost
QDRANT_PORT=6333
```

**CHECKPOINT:** `.env` exists with a working provider (or Ollama running).

---

## PHASE 1: Infrastructure (Docker)

### Step 1.1 — Qdrant only

```powershell
docker compose up -d qdrant
```

```powershell
Invoke-RestMethod http://localhost:6333/healthz
```

**CHECKPOINT:** Qdrant responds on port 6333.

### Step 1.2 — Langfuse (optional)

```powershell
docker compose up -d langfuse-db langfuse-server
```

Open `http://localhost:3000`. Failures here are OK for core RAG.

---

## PHASE 2: Config & Import Smoke Test

### Step 2.1 — Config loads

```powershell
python -c "from medflow.config import get_settings; s = get_settings(); print(s.model_dump_json(indent=2))"
```

**CHECKPOINT:** JSON prints with `llm`, `embedding`, `qdrant`, etc.

### Step 2.2 — Provider / pipeline imports

```powershell
python -c "from medflow.providers.factory import ProviderFactory; print('Providers OK')"
python -c "from medflow.ocr.pipeline import OCRPipeline; print('OCR OK')"
python -c "from medflow.deidentify.pipeline import DeIDPipeline; print('DeID OK')"
python -c "from medflow.retrieval.pipeline import RetrievalPipeline; print('Retrieval OK')"
python -c "from medflow.generation.pipeline import GenerationPipeline; print('Generation OK')"
python -c "from medflow.evaluation.runner import EvaluationRunner; print('Evaluation OK')"
```

**CHECKPOINT:** All six succeed (fix any `ModuleNotFoundError` with `pip install …`).

---

## PHASE 3: Synthetic Data

### Step 3.1 — Verify corpus

```powershell
dir data\synthetic\documents\
```

Expect ~24 PDFs (+ matching `.txt` / `.json` sidecars).

```powershell
python -c "import json; d=json.load(open('data/synthetic/golden_qa.json', encoding='utf-8')); print(f'{len(d)} Q&A pairs')"
```

If missing:

```powershell
python data/synthetic/generate_synthetic.py
```

**CHECKPOINT:** Documents + `golden_qa.json` exist.

---

## PHASE 4: API Startup

### Step 4.1 — Start API

```powershell
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Watch for Qdrant connection errors if Docker is not running.

### Step 4.2 — Health

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/ready
```

Expect `status: healthy` and `status: ready` when Qdrant is reachable.

### Step 4.3 — OpenAPI

[http://localhost:8000/docs](http://localhost:8000/docs)

**CHECKPOINT:** Swagger lists upload, documents, query, evaluate, health, ready.

---

## PHASE 5: Core Functionality

### Step 5.1 — Upload (correct filename)

```powershell
curl -X POST http://localhost:8000/api/v1/documents/upload -F "file=@data/synthetic/documents/doc_prior_auth_001.pdf"
```

PowerShell:

```powershell
$form = @{ file = Get-Item "data\synthetic\documents\doc_prior_auth_001.pdf" }
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/documents/upload" -Method Post -Form $form
```

Expect `doc_id`, `classification`, `chunk_count`, `processing_time_ms`.

### Step 5.2 — List documents

```powershell
Invoke-RestMethod http://localhost:8000/api/v1/documents
```

### Step 5.3 — Query

```powershell
$body = @{ query = "What CPT code is requested in the prior authorization?" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/query" -Method Post -Body $body -ContentType "application/json; charset=utf-8"
```

Answers cite context like `[Doc-<source_doc_id>, Chunk-<n>]` when the model follows the system prompt.

**CHECKPOINT:** Upload → list → query works end-to-end.

---

## PHASE 6: Seed & Evaluation

### Step 6.1 — Seed all PDFs

```powershell
python scripts/seed_synthetic_data.py
```

Or `make seed` from repo root.

### Step 6.2 — Evaluation

```powershell
python scripts/run_evaluation.py --profile default
# or
make evaluate
```

Uses many LLM calls; prefer **Ollama** in dev or budget API spend.

**CHECKPOINT:** `evaluation_results/eval_*.json` written.

---

## PHASE 7: Dashboard

```powershell
streamlit run src/dashboard/app.py
```

Or `make dashboard`. Set `MEDFLOW_API_BASE=http://127.0.0.1:8000` if the API is not on localhost.

Verify all four sidebar pages load at [http://localhost:8501](http://localhost:8501).

---

## PHASE 8: Tests

### Step 8.1 — Unit tests (no live Qdrant required)

```powershell
pytest tests/unit/ -v
```

Full suite with coverage:

```powershell
make test
```

(`make test` adds `--cov=medflow --cov=api`.)

### Step 8.2 — Integration tests

```powershell
pytest tests/integration/ -v
```

Most tests **mock** external services; if a test fails on collection, read the traceback—`tests/integration/test_api_endpoints.py` uses `lifespan="off"` and does not require Qdrant.

---

## PHASE 9: Docker full stack

```powershell
docker compose up -d --build
```

Verify API `:8000`, dashboard `:8501`, Qdrant `:6333`, Langfuse `:3000` as configured.

---

## Results tracker

```
PHASE 0: Environment
[ ] Python 3.11 in venv
[ ] pip install -e ".[dev,dashboard]" succeeded
[ ] .env / Ollama configured

PHASE 1: Infrastructure
[ ] Qdrant :6333
[ ] Langfuse :3000 (optional)

PHASE 2: Imports
[ ] get_settings() OK
[ ] All 6 import one-liners OK

PHASE 3: Data
[ ] ~24 synthetic documents
[ ] golden_qa.json present

PHASE 4: API
[ ] uvicorn starts
[ ] /health -> healthy
[ ] /docs loads

PHASE 5: Core
[ ] POST .../documents/upload OK
[ ] GET .../documents OK
[ ] POST .../query OK

PHASE 6: Evaluation
[ ] seed completed
[ ] evaluation JSON produced

PHASE 7: Dashboard
[ ] Four pages render

PHASE 8: Tests
[ ] Unit: __/__ 
[ ] Integration: __/__ 

PHASE 9: Docker
[ ] compose up --build OK
```

---

## What to report when asking for help

1. Filled checklist  
2. Full traceback (not only the last line)  
3. Provider: OpenAI / Anthropic / Ollama  
