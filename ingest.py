import argparse
from pathlib import Path
from typing import Any

from typing_extensions import TypedDict

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_DIMENSIONS,
    DEFAULT_PERSIST_DIR,
)


class IngestionState(TypedDict, total=False):
    file_path: str
    persist_dir: str
    chunk_size: int
    chunk_overlap: int
    documents: list[Any]
    chunks: list[Any]
    embeddings: Any
    vectorstore: Any


def build_embeddings():
    from embeddings import LocalHashEmbeddings

    return LocalHashEmbeddings(dimensions=EMBEDDING_DIMENSIONS)


def load_pdf_node(state: IngestionState) -> IngestionState:
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(state["file_path"])
    return {**state, "documents": loader.load()}


def split_node(state: IngestionState) -> IngestionState:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=state["chunk_size"],
        chunk_overlap=state["chunk_overlap"],
        add_start_index=True,
    )
    return {**state, "chunks": splitter.split_documents(state["documents"])}


def embed_node(state: IngestionState) -> IngestionState:
    return {**state, "embeddings": build_embeddings()}


def store_node(state: IngestionState) -> IngestionState:
    from langchain_community.vectorstores import FAISS

    persist_dir = Path(state["persist_dir"])
    persist_dir.mkdir(parents=True, exist_ok=True)

    vectorstore = FAISS.from_documents(
        documents=state["chunks"],
        embedding=state["embeddings"],
    )
    vectorstore.save_local(str(persist_dir))

    return {**state, "vectorstore": vectorstore}


def build_ingestion_graph():
    from langgraph.graph import END, StateGraph

    graph = StateGraph(IngestionState)
    graph.add_node("load", load_pdf_node)
    graph.add_node("split", split_node)
    graph.add_node("embed", embed_node)
    graph.add_node("store", store_node)
    graph.set_entry_point("load")
    graph.add_edge("load", "split")
    graph.add_edge("split", "embed")
    graph.add_edge("embed", "store")
    graph.add_edge("store", END)
    return graph.compile()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a PDF, split it into chunks, embed the chunks, and persist them to Chroma."
    )
    parser.add_argument("--file", required=True, help="Path to the PDF file to ingest.")
    parser.add_argument(
        "--persist-dir",
        default=str(DEFAULT_PERSIST_DIR),
        help="Directory where the vector store will be persisted.",
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument(
        "--show-graph",
        action="store_true",
        help="Print the LangGraph node flow before running ingestion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.file)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if args.show_graph:
        print("LangGraph ingestion nodes: load -> split -> embed -> store")

    graph = build_ingestion_graph()
    result = graph.invoke(
        {
            "file_path": str(pdf_path),
            "persist_dir": args.persist_dir,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
        }
    )

    print(f"Loaded pages: {len(result['documents'])}")
    print(f"Stored chunks: {len(result['chunks'])}")
    print(f"Persist directory: {Path(args.persist_dir).resolve()}")


if __name__ == "__main__":
    main()
