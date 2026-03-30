"""Synthetic PDFs + golden QA (OpenAI). Needs OPENAI_API_KEY; run from repo root."""

from __future__ import annotations

import json
import os
import random
import re
import sys
import textwrap
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

load_dotenv(_ROOT / ".env")

MODEL = os.environ.get("MEDFLOW_SYNTH_MODEL", "gpt-4o-mini")
CATEGORIES = [
    "PRIOR_AUTH",
    "REFERRAL",
    "RECORDS_REQUEST",
    "LAB_RESULTS",
    "INSURANCE",
    "OTHER",
]
DOCS_PER_CATEGORY = 4


def _client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("ERROR: Set OPENAI_API_KEY in .env or environment.", file=sys.stderr)
        raise SystemExit(2)
    return OpenAI(api_key=key)


def generate_document_body(client: OpenAI, category: str, doc_index: int) -> tuple[str, dict]:
    """Ask the model for a realistic 1–2 page clinical/administrative document."""
    prompt = f"""You are generating synthetic training data for a healthcare NLP system (not real PHI).

Write a realistic {category.replace('_', ' ')} document as plain text with clear section headers.
Include realistic-looking but fictional: patient name, MRN format, DOB, address line, phone,
insurance member ID / group where relevant, provider name and NPI-style 10-digit number,
at least one ICD-10 code (valid format like M54.5 or E11.9), at least one CPT or HCPCS code
where clinically appropriate, dates in ISO or US format, and clinical or administrative narrative
that would appear on real paperwork. Length: 400–900 words. No boilerplate about being an AI."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You write realistic U.S. healthcare document text only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.65,
        max_tokens=2500,
    )
    text = (resp.choices[0].message.content or "").strip()
    meta = {
        "category": category,
        "doc_index": doc_index,
        "model": MODEL,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tokens": resp.usage.total_tokens if resp.usage else None,
    }
    return text, meta


def write_pdf(path: Path, title: str, body: str) -> None:
    """Render multi-page PDF with wrapped lines."""
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    c.setTitle(title)
    y = height - inch
    margin_left = inch
    margin_right = width - inch
    line_height = 12
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin_left, y, title[:100])
    y -= 0.4 * inch
    c.setFont("Helvetica", 9)
    wrapped = textwrap.fill(body, width=95).split("\n")
    for line in wrapped:
        if y < inch:
            c.showPage()
            y = height - inch
            c.setFont("Helvetica", 9)
        c.drawString(margin_left, y, line[:120])
        y -= line_height
    c.save()


def generate_qa_for_doc(client: OpenAI, doc_id: str, category: str, text: str) -> list[dict]:
    """Generate 2 grounded Q&A pairs plus optional difficulty labels."""
    excerpt = text[:12000]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "Return ONLY valid JSON: an array of objects with keys: question, ground_truth, difficulty (easy|medium|hard).",
            },
            {
                "role": "user",
                "content": f"""Document id: {doc_id}  Category: {category}

Document text:
{excerpt}

Create exactly 2 question/answer pairs whose answers are fully supported by the document text.
Questions should require different facts. ground_truth must be complete sentences.""",
            },
        ],
        temperature=0.3,
        max_tokens=800,
    )
    raw = (resp.choices[0].message.content or "").strip()
    m = re.search(r"\[[\s\S]*\]", raw)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out: list[dict] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        gt = str(item.get("ground_truth", "")).strip()
        diff = str(item.get("difficulty", "medium"))
        if q and gt:
            out.append(
                {
                    "question": q,
                    "ground_truth": gt,
                    "difficulty": diff,
                },
            )
    return out


def generate_unanswerable(client: OpenAI, n: int = 8) -> list[dict]:
    """Questions whose answers are not in any synthetic chart (abstention test)."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "Return ONLY a JSON array of {question, ground_truth} objects.",
            },
            {
                "role": "user",
                "content": f"""Write {n} questions about medical records that are NOT answerable without external knowledge
(e.g. patient favorite color, unrelated hospital policy, future events). ground_truth should say the information is not in the provided documents.""",
            },
        ],
        temperature=0.5,
        max_tokens=1200,
    )
    raw = (resp.choices[0].message.content or "").strip()
    m = re.search(r"\[[\s\S]*\]", raw)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out: list[dict] = []
    for item in arr:
        if isinstance(item, dict) and item.get("question"):
            out.append(
                {
                    "question": str(item["question"]),
                    "ground_truth": str(item.get("ground_truth", "Information is not contained in the document corpus.")),
                    "difficulty": "hard",
                },
            )
    return out[:n]


def main() -> None:
    random.seed(42)
    client = _client()
    root = Path(__file__).resolve().parent
    doc_dir = root / "documents"
    doc_dir.mkdir(parents=True, exist_ok=True)

    all_meta: list[dict] = []
    qa: list[dict] = []
    qid = 1

    for cat in CATEGORIES:
        for j in range(DOCS_PER_CATEGORY):
            idx = len(all_meta) + 1
            doc_id = f"doc_{cat.lower()}_{idx:03d}"
            print(f"Generating {doc_id} ...", flush=True)
            body, meta = generate_document_body(client, cat, idx)
            meta["doc_id"] = doc_id
            meta["category"] = cat
            txt_path = doc_dir / f"{doc_id}.txt"
            pdf_path = doc_dir / f"{doc_id}.pdf"
            json_path = doc_dir / f"{doc_id}.json"
            txt_path.write_text(body + "\n", encoding="utf-8")
            write_pdf(pdf_path, doc_id.replace("_", " ").title(), body)
            json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            all_meta.append(meta)

            pairs = generate_qa_for_doc(client, doc_id, cat, body)
            for p in pairs:
                qa.append(
                    {
                        "id": f"qa_{qid:03d}",
                        "question": p["question"],
                        "ground_truth": p["ground_truth"],
                        "source_doc_ids": [doc_id],
                        "category": cat,
                        "difficulty": p.get("difficulty", "medium"),
                    },
                )
                qid += 1

    need = max(0, 52 - len(qa))
    if need > 0:
        for u in generate_unanswerable(client, min(need + 4, 16)):
            qa.append(
                {
                    "id": f"qa_{qid:03d}",
                    "question": u["question"],
                    "ground_truth": u["ground_truth"],
                    "source_doc_ids": [],
                    "category": "OTHER",
                    "difficulty": "hard",
                },
            )
            qid += 1
            if len(qa) >= 52:
                break

    (root / "golden_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_meta)} document sets and {len(qa)} golden Q&A rows under {doc_dir.parent}.")


if __name__ == "__main__":
    main()
