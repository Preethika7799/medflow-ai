# Runbook (copy-paste)

Clone path may be `Medflow_ai` or `medflow-ai` — fix `cd` accordingly.

## Needs

- Python 3.11
- `OPENAI_API_KEY` in `.env` at repo root
- Optional: Qdrant Cloud — `MEDFLOW_QDRANT__URL` and `MEDFLOW_QDRANT__API_KEY` in `.env` (see `docs/testing-playbook.md`)

---

## STEP 1: Fresh venv + install (5–10 min)

```powershell
cd C:\path\to\Medflow_ai
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install spacy
python -m spacy download en_core_web_lg

pip install paddlepaddle==2.6.2
pip install paddleocr

pip install reportlab

pip install -e ".[dev,dashboard,notebooks]"
```

---

## STEP 2: Verify install

```powershell
python -c "from medflow.config import get_settings; s = get_settings(); print(f'Provider: {s.llm.provider}, Model: {s.llm.model}')"
python -c "from medflow.providers.factory import ProviderFactory; print('Providers OK')"
python -c "from medflow.ocr.pipeline import OCRPipeline; print('OCR OK')"
python -c "from medflow.deidentify.pipeline import DeIDPipeline; print('DeID OK')"
python -c "from medflow.retrieval.pipeline import RetrievalPipeline; print('Retrieval OK')"
python -c "from medflow.evaluation.runner import EvaluationRunner; print('Evaluation OK')"
```

---

## STEP 3: Test OpenAI (use full settings, not `s.llm`)

`ProviderFactory.create` / `create_llm` expect **`MedFlowSettings`**, not `settings.llm`.

```powershell
python -c "
import asyncio
from medflow.config import get_settings
from medflow.providers.factory import ProviderFactory

async def test():
    s = get_settings()
    p = ProviderFactory.create(s)
    r = await p.chat([{'role': 'user', 'content': 'Say hello in exactly 5 words.'}])
    print('Response:', r.content)
    print('Tokens:', r.usage)
    print('OpenAI WORKING')

asyncio.run(test())
"
```

---

## STEP 4: Generate synthetic data

```powershell
python data/synthetic/generate_synthetic.py
```

```powershell
(Get-ChildItem data\synthetic\documents\*.pdf).Count
python -c "import json; d=json.load(open('data/synthetic/golden_qa.json', encoding='utf-8')); print(len(d), 'QA pairs')"
```

---

## STEP 5: Qdrant

**Local Docker:**

```powershell
docker compose up -d qdrant
Start-Sleep 10
curl.exe http://localhost:6333/healthz
```

**Or standalone:**

```powershell
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

**Qdrant Cloud:** configure `.env` with `MEDFLOW_QDRANT__URL` + `MEDFLOW_QDRANT__API_KEY`; skip local Docker.

---

## STEP 6: Seed

```powershell
python scripts/seed_synthetic_data.py
```

Verify (local Qdrant):

```powershell
python -c "
from qdrant_client import QdrantClient
from medflow.config import get_settings
from medflow.qdrant_utils import build_sync_qdrant_client
c = build_sync_qdrant_client(get_settings())
for col in c.get_collections().collections:
    info = c.get_collection(col.name)
    print(f'{col.name}: {info.points_count} vectors')
"
```

---

## STEP 7: API

Terminal 1:

```powershell
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2:

```powershell
curl.exe http://localhost:8000/health

$pdf = (Get-ChildItem data\synthetic\documents\*.pdf | Select-Object -First 1).FullName
curl.exe -X POST http://localhost:8000/api/v1/documents/upload -F "file=@$pdf"

curl.exe -X POST http://localhost:8000/api/v1/query -H "Content-Type: application/json" -d "{\"query\":\"What procedure is requested in the prior authorization?\"}"

curl.exe http://localhost:8000/api/v1/documents
```

---

## STEP 8: Evaluation

```powershell
python scripts/run_evaluation.py
```

---

## STEP 9: Dashboard

```powershell
streamlit run src/dashboard/app.py
```

Open http://localhost:8501

---

## STEP 10: Tests

```powershell
pytest tests/unit/ -v --tb=short
```

---

## STEP 11: Notebooks (optional)

```powershell
jupyter notebook notebooks/finetune_classifier/01_generate_training_data.ipynb
```

Notebooks **02–03** are heavy on CPU; **Colab + GPU** is recommended for full LoRA runs.

---

## Report back checklist

1. First failing step + full traceback (if any)  
2. Evaluation aggregate lines from Step 8  
3. `pytest` pass/fail counts  
4. Dashboard: real API data yes/no  
