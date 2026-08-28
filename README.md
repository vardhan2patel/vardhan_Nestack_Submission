# PDF Vectorisation Pipeline

Python implementation of the Nestack LangChain vectorisation assessment. The project ingests a PDF into a persisted Chroma vector store and exposes a single retrieval endpoint that returns raw similarity-search chunks.

## Requirements Covered

- `python ingest.py --file document.pdf` loads, splits, embeds, and persists a PDF.
- LangChain document loader: `PyPDFLoader`.
- LangChain text splitter: `RecursiveCharacterTextSplitter`.
- LangChain embedding primitive: local deterministic `Embeddings` implementation.
- LangChain-compatible vector store: `FAISS`.
- Retrieval endpoint: `POST /query`.
- No `RetrievalQA`, `ConversationalRetrievalChain`, QA chain, conversational chain, or LLM call is used.
- LangGraph models ingestion as named nodes: `load -> split -> embed -> store`.

## Project Structure

```text
.
├── config.py
├── embeddings.py
├── ingest.py
├── server.py
├── requirements.txt
├── results.json
├── README.md
└── vector_store/          # Created after ingestion; excluded from git
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Ingest A PDF

```bash
python ingest.py --file document.pdf --show-graph
```

Expected console output includes:

```text
LangGraph ingestion nodes: load -> split -> embed -> store
Loaded pages: 2
Stored chunks: 4
Persist directory: .../vector_store
```

## Run Retrieval Server

```bash
uvicorn server:app --reload
```

Query the vector store:

```bash
curl -X POST http://127.0.0.1:8000/query ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What should the retrieval endpoint return?\",\"top_k\":3}"
```

Response format:

```json
[
  {
    "chunk_text": "matching chunk text",
    "page_number": 1,
    "score": 0.42
  }
]
```

## Component Choices

`PyPDFLoader` is used because the assessment specifically asks for a LangChain PDF document loader, and PyPDFLoader preserves per-page metadata that the API can return as `page_number`.

`RecursiveCharacterTextSplitter` is used with `chunk_size=650` and `chunk_overlap=100`. A 650-character chunk keeps requirement paragraphs focused enough for retrieval, while 100 characters of overlap preserves context across chunk boundaries without creating too much duplicate text.

`LocalHashEmbeddings` implements LangChain's `Embeddings` interface and is used because it runs locally, is deterministic, and avoids API keys or model downloads. It converts normalized tokens into a fixed-size hashed vector, which is appropriate for this small assessment PDF and keeps ingestion/retrieval reproducible.

`FAISS` is used because it is LangChain-compatible, persists locally with `save_local()`, reloads with `load_local()`, and supports direct `similarity_search_with_score()` calls. The server calls that method directly and returns raw chunk results.

## LangGraph Flow

```text
load -> split -> embed -> store
```

- `load`: reads the PDF with `PyPDFLoader`.
- `split`: chunks documents with `RecursiveCharacterTextSplitter`.
- `embed`: creates the HuggingFace embedding function.
- `store`: writes chunks and embeddings to persisted Chroma storage.

## Sample Results

`results.json` contains three sample query outputs in the required response format:

- "What are the ingestion steps?"
- "What should the retrieval endpoint return?"
- "Which chains are not allowed?"

## Deployment Link

Add the deployed API URL here before submission:

```text
https://your-deployment-url.example.com
```

## Submission Notes

Create a private GitHub repository named `{yourName_Nestack_Submission}` and add the required contributors listed in the assessment. Public repositories are disqualified.
