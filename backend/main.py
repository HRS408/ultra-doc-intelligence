"""FastAPI application: API routes, startup embedder, frontend mount."""

import asyncio
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from groq import AuthenticationError, GroqError, RateLimitError

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

from backend.extraction import extract_structured
from backend.ingestion import build_chunk_rows, embed_and_store, get_chroma_client, parse_document, stable_doc_id
from backend.retrieval import run_ask
from backend.schemas import AskRequest, AskResponse, ExtractRequest, ExtractResponse, HealthResponse, UploadResponse

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

FRONTEND_DIR = str(_PROJECT_ROOT / "frontend")

app = FastAPI(title="Ultra Doc-Intelligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    root = str(_PROJECT_ROOT)
    os.makedirs(os.path.join(root, "storage", "chroma"), exist_ok=True)

    app.state.embedder = None  # lazy load
    app.state.chroma = get_chroma_client()

def get_embedder(app):
    if app.state.embedder is None:
        from sentence_transformers import SentenceTransformer
        app.state.embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return app.state.embedder

def _embed_list_sync(model, texts: List[str]) -> List[List[float]]:
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return emb.tolist()


def _collection_or_404(request: Request, doc_id: str):
    name = f"doc_{doc_id}"
    try:
        return request.app.state.chroma.get_collection(name)
    except Exception:
        raise HTTPException(status_code=404, detail="Unknown doc_id or document not indexed") from None


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


def _sanitize_token(s: str, default: str, max_len: int = 64) -> str:
    t = (s or "").strip() or default
    return t[:max_len]


@app.post("/upload", response_model=UploadResponse)
async def upload(
    request: Request,
    file: UploadFile = File(...),
    doc_type: str = Form("general"),
    load_id: str = Form(""),
) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    lower = file.filename.lower()
    if not (lower.endswith(".pdf") or lower.endswith(".docx") or lower.endswith(".txt")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use PDF, DOCX, or TXT.",
        )

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds 10MB limit")

    filename = file.filename
    doc_id = stable_doc_id(raw)
    dt = _sanitize_token(doc_type, "general")
    lid_resolved = _sanitize_token(load_id, doc_id, max_len=128)
    embedder = get_embedder(request.app)
    chroma_client = request.app.state.chroma

    def _work() -> int:
        segments = parse_document(raw, filename)
        rows = build_chunk_rows(segments, dt, lid_resolved)
        return embed_and_store(
            doc_id,
            os.path.basename(filename),
            rows,
            lambda texts: _embed_list_sync(embedder, texts),
            chroma_client,
        )

    loop = asyncio.get_event_loop()
    try:
        chunk_count = await loop.run_in_executor(None, _work)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return UploadResponse(
        doc_id=doc_id,
        filename=filename,
        chunk_count=chunk_count,
        status="indexed",
        doc_type=dt,
        load_id=lid_resolved,
    )


@app.post("/ask", response_model=AskResponse)
async def ask(request: Request, body: AskRequest) -> AskResponse:
    col = _collection_or_404(request, body.doc_id)
    embedder = request.app.state.embedder
    chroma_client = request.app.state.chroma

    def _embed_fn(texts: List[str]) -> List[List[float]]:
        return _embed_list_sync(embedder, texts)
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(
            None,
            lambda: run_ask(col, chroma_client, body.question, _embed_fn, embedder),
        )
    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Rate limit reached, retry in a few seconds.",
        ) from None
    except AuthenticationError:
        raise HTTPException(
            status_code=401,
            detail=(
                "Groq rejected the API key (invalid or still set to the .env.example placeholder). "
                "Create a key at https://console.groq.com/keys and set GROQ_API_KEY in .env in the project root."
            ),
        ) from None
    except GroqError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Groq API error: {getattr(e, 'message', None) or str(e)}",
        ) from e


@app.post("/extract", response_model=ExtractResponse)
async def extract(request: Request, body: ExtractRequest) -> ExtractResponse:
    col = _collection_or_404(request, body.doc_id)

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, lambda: extract_structured(col))
    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Rate limit reached, retry in a few seconds.",
        ) from None
    except AuthenticationError:
        raise HTTPException(
            status_code=401,
            detail=(
                "Groq rejected the API key (invalid or still set to the .env.example placeholder). "
                "Create a key at https://console.groq.com/keys and set GROQ_API_KEY in .env in the project root."
            ),
        ) from None
    except GroqError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Groq API error: {getattr(e, 'message', None) or str(e)}",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
