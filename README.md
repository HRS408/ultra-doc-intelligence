# Ultra Doc-Intelligence

Ultra Doc-Intelligence is a RAG-based logistics document Q&A system. It parses PDFs, DOCX files, and TXT files, indexes the content in ChromaDB, and answers shipment questions using Groq's `llama-3.3-70b-versatile` model with strict document-only guardrails.

Example questions:

- What is the Load ID?
- Who is the consignee?
- What is the PO number?
- What is the COD value?

The system is optimized for logistics PDFs such as Bills of Lading and rate confirmations, where important values often appear as compact label/value pairs.

## Architecture

The pipeline is:

```text
Document
  -> Parsing
  -> Structured Extraction
  -> Chunking
  -> Embedding
  -> Storage (ChromaDB)
  -> Retrieval
  -> LLM Answering
  -> Confidence Scoring
```

At upload time, the backend creates a stable document id from the file bytes, parses the document, extracts known logistics fields, builds retrieval chunks, embeds those chunks with `all-MiniLM-L6-v2`, and stores them in a ChromaDB collection named `doc_{doc_id}`.

At question time, the system retrieves relevant chunks, builds a small context window, asks Groq to answer only from that context, and computes confidence from answer/evidence alignment.

## Parsing Strategy

PDFs are parsed with PyMuPDF (`fitz`) using block-level extraction:

```python
page.get_text("blocks")
```

Raw `get_text()` output can flatten a logistics PDF into an order that is visually wrong. Labels and values may be separated across lines, tables may lose their structure, and similarity search becomes unreliable.

Block-level parsing keeps layout coordinates. Blocks are sorted top-to-bottom by vertical position and then left-to-right by horizontal position. Blocks on the same visual row are reconstructed with separators so label/value pairs remain readable.

This helps preserve rows such as:

```text
Load ID LD53657
PO Number Pickup 112233ABC
COD Value $64000 USD
```

## Chunking Strategy

The project uses three complementary chunk types.

**Section chunks** preserve the original document flow. The section chunker detects all-caps headers and lines ending in `:`, keeps moderate sections intact, and recursively splits long sections.

**Structured summary chunks** collect extracted logistics fields into a compact summary:

```text
Load ID: LD53657
PO Number: 112233ABC
Consignee: xyz
COD Value: $64000 USD
```

**Atomic field-level chunks** are the most important retrieval improvement. Each extracted field is stored as a standalone chunk:

```text
[FIELD] Load ID: LD53657
[FIELD] Consignee: xyz
```

Atomic chunks make common questions deterministic. A query like "Who is the consignee?" maps directly to a chunk containing only the relevant field instead of competing with a noisy full-page table.

## Embedding & Storage

Embeddings use `sentence-transformers` with `all-MiniLM-L6-v2`.

MiniLM is small enough to run locally, fast enough for interactive uploads, and strong enough for short logistics field retrieval when paired with structured chunks. It also keeps the application practical for a Windows executable build.

ChromaDB stores each uploaded document in a persistent local collection. By default, runtime data is written to:

```text
storage/chroma
```

In source mode, this directory is under the project root. In the packaged Windows build, it is created beside the `.exe`.

## Retrieval Method

Retrieval keeps the existing ranking behavior:

- vector similarity uses cosine distance from ChromaDB
- top-k retrieval uses `k=5`
- chunks below similarity `0.30` are filtered out
- the top three passing chunks are passed to the LLM as context

The project also includes a lightweight hybrid layer using BM25 plus Chroma vector retrieval with weighted rank fusion. Metadata prefilters can narrow questions to likely document types, then fall back to the full document if no chunks are found.

## Guardrails

The LLM system prompt requires answers to come only from the retrieved context. If the answer is not present, the model must return:

```text
NOT_IN_DOCUMENT
```

Guardrails trigger when:

- no retrieved chunk passes the relevance floor
- the model returns `NOT_IN_DOCUMENT`
- confidence is low after evidence scoring

Low-confidence answers are replaced with a safe response rather than presenting unsupported text. This reduces hallucination risk while still allowing high-confidence field answers to pass through.

## Confidence Scoring

Confidence is based on answer/evidence alignment rather than raw retrieval similarity alone.

The score combines:

- exact answer match in retrieved context
- a field-level boost when the answer appears in a `[FIELD]` chunk
- a smaller contribution from the best retrieval similarity
- optional LLM verification when exact structured grounding is missing

This is better than raw similarity because a chunk can be semantically relevant without proving the final answer. Conversely, an atomic chunk such as `[FIELD] Load ID: LD53657` is very strong evidence for the answer `LD53657`, even if the similarity score is only moderate.

Expected behavior:

- exact field matches: high confidence, usually `0.85-0.95`
- partial support: medium/high confidence, usually `0.60-0.75`
- weak or unsupported answers: low confidence, usually below `0.40`

## Failure Cases

Known limitations:

- scanned PDFs with no text layer are not OCR'd
- badly formatted PDFs can still produce noisy block order
- missing fields cannot be recovered from retrieval alone
- handwritten documents are unsupported without OCR
- ambiguous questions may retrieve multiple plausible fields
- Groq API errors or rate limits can interrupt answer generation

## Future Improvements

Useful next steps:

- LLM-based extraction as a fallback when regex extraction misses fields
- stronger metadata filters for document type and field type
- cross-encoder reranking over retrieved chunks
- OCR support for scanned Bills of Lading
- batch ingestion for many documents
- multi-document search and comparison
- richer UI states for sources, confidence, and field extraction

## How to Run

Prerequisites:

- Python 3.11+
- a Groq API key

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local environment file:

```bash
copy .env.example .env
```

Set:

```text
GROQ_API_KEY=your_key_here
```

Start the app:

```bash
uvicorn backend.main:app --reload --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

Upload a logistics PDF, then ask questions such as:

```text
What is the Load ID?
Who is the consignee?
```

The first run may download the `all-MiniLM-L6-v2` model. Later runs use the local cache.

## Notes

Runtime data is stored under `storage/chroma` relative to the project root. Keep `.env` local and do not commit real API keys.
