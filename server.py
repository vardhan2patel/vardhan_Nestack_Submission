from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import DEFAULT_PERSIST_DIR
from ingest import build_embeddings


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=20)


class QueryResult(BaseModel):
    chunk_text: str
    page_number: int | None
    score: float


app = FastAPI(title="PDF Vectorisation Retrieval API")


@lru_cache(maxsize=1)
def get_vectorstore():
    from langchain_community.vectorstores import FAISS

    persist_dir = Path(DEFAULT_PERSIST_DIR)
    if not persist_dir.exists():
        raise RuntimeError(
            f"Vector store not found at {persist_dir}. Run ingest.py before starting queries."
        )

    return FAISS.load_local(
        str(persist_dir),
        build_embeddings(),
        allow_dangerous_deserialization=True,
    )


@app.post("/query", response_model=list[QueryResult])
def query_store(payload: QueryRequest) -> list[QueryResult]:
    try:
        vectorstore = get_vectorstore()
        matches = vectorstore.similarity_search_with_score(
            payload.query,
            k=payload.top_k,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    results: list[QueryResult] = []
    for document, score in matches:
        raw_page = document.metadata.get("page")
        page_number = raw_page + 1 if isinstance(raw_page, int) else None
        results.append(
            QueryResult(
                chunk_text=document.page_content,
                page_number=page_number,
                score=float(score),
            )
        )

    return results
