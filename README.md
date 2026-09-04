# PDF Vectorisation Pipeline

Python implementation of the Nestack LangChain vectorisation assessment. The project ingests a PDF, processes it into vector embeddings, stores them in a persisted FAISS vector store, and exposes a retrieval endpoint that returns raw similarity-search chunks.

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

Create a virtual environment:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### macOS/Linux

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Ingest a PDF

Run the ingestion pipeline using:

```bash
python ingest.py --file document.pdf --show-graph
```

The ingestion pipeline follows:

```text
PDF
 ↓
Load
 ↓
Split
 ↓
Embed
 ↓
Store
```

Expected console output includes:

```text
LangGraph ingestion nodes: load -> split -> embed -> store
Loaded pages: 2
Stored chunks: 4
Persist directory: .../vector_store
```

## Run Retrieval Server

Start the FastAPI server:

```bash
uvicorn server:app --reload
```

The server will start locally at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Query the Vector Store

Send a POST request to the `/query` endpoint:

```bash
curl -X POST http://127.0.0.1:8000/query ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What should the retrieval endpoint return?\",\"top_k\":3}"
```

Example request body:

```json
{
  "query": "What are the ingestion steps?",
  "top_k": 3
}
```

## Response Format

```json
[
  {
    "chunk_text": "matching chunk text",
    "page_number": 1,
    "score": 0.42
  }
]
```

The response contains:

- `chunk_text`: The relevant text chunk retrieved from the PDF.
- `page_number`: The PDF page from which the chunk originated.
- `score`: The distance/score returned by FAISS for the retrieved result.

## Component Choices

### PyPDFLoader

`PyPDFLoader` is used because the assessment specifically requires a LangChain PDF document loader. It extracts the PDF content while preserving page metadata, allowing the API to return the corresponding `page_number` for each retrieved chunk.

### RecursiveCharacterTextSplitter

`RecursiveCharacterTextSplitter` is used with:

```text
chunk_size = 650
chunk_overlap = 100
```

Large PDF documents are divided into smaller chunks before embedding.

The chunk overlap helps preserve context between consecutive chunks and reduces the possibility of losing information at chunk boundaries.

### LocalHashEmbeddings

`LocalHashEmbeddings` is a custom implementation of LangChain's `Embeddings` interface.

It is used because it:

- Runs completely locally.
- Does not require API keys.
- Does not require internet access.
- Does not require downloading external models.
- Produces deterministic embeddings for the same input.

The embedding implementation converts normalized text tokens and token bigrams into a fixed-size hashed vector.

For a production-level semantic search system, a pretrained embedding model could be used to provide stronger semantic understanding. The local implementation was chosen to keep this assessment lightweight, reproducible, and self-contained.

### FAISS

`FAISS` is used as the vector store because it:

- Is compatible with LangChain.
- Supports efficient vector similarity search.
- Can persist the vector index locally using `save_local()`.
- Can reload the vector store using `load_local()`.
- Supports direct `similarity_search_with_score()` calls.

The retrieval server directly performs similarity search and returns the raw matching document chunks without using an LLM or QA chain.

## LangGraph Flow

The ingestion workflow is implemented using LangGraph.

```text
load -> split -> embed -> store
```

- `load`: Reads the PDF using `PyPDFLoader`.
- `split`: Divides the PDF content into smaller chunks using `RecursiveCharacterTextSplitter`.
- `embed`: Creates the `LocalHashEmbeddings` implementation used to convert document chunks into vector representations.
- `store`: Stores the document chunks and their embeddings in a persisted FAISS vector store.

## Retrieval Flow

Once the PDF has been ingested and stored, the retrieval process works as follows:

```text
User Query
     ↓
FastAPI /query Endpoint
     ↓
Convert Query into Embedding
     ↓
FAISS Similarity Search
     ↓
Retrieve Top K Matching Chunks
     ↓
Return Chunk Text + Page Number + Score
```

The same embedding implementation is used for both document chunks and user queries so that they exist in the same vector space for comparison.

## API Validation

The `/query` endpoint uses Pydantic models for request validation.

Example request:

```json
{
  "query": "What are embeddings?",
  "top_k": 3
}
```

Validation rules:

- `query` cannot be empty.
- `top_k` must be between 1 and 20.

Invalid requests automatically return a `422 Validation Error`.

## Sample Results

`results.json` contains sample query outputs in the required response format for:

- "What are the ingestion steps?"
- "What should the retrieval endpoint return?"
- "Which chains are not allowed?"

## Deployment

The API is deployed and accessible at:

https://vardhan-nestack-submission.onrender.com/docs

## Key Design Decisions

- The project uses a retrieval-only architecture.
- No LLM is used.
- No answer-generation chain is used.
- No `RetrievalQA` or `ConversationalRetrievalChain` is used.
- FAISS performs direct vector similarity search.
- FastAPI exposes the retrieval functionality through a REST API.
- LangGraph structures the ingestion pipeline into modular processing stages.

## Submission Notes

The repository is maintained as a private GitHub repository according to the assessment requirements.

The ingestion pipeline, retrieval endpoint, sample results, and deployment documentation are included in the submission.
