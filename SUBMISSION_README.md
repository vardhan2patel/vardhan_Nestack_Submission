# Nestack Assessment Submission

This repository is structured around the required Python LangChain PDF vectorisation pipeline.

Primary assessment files:

- `ingest.py` - LangGraph ingestion script runnable as `python ingest.py --file document.pdf`
- `server.py` - FastAPI server exposing `POST /query`
- `embeddings.py` - local LangChain `Embeddings` implementation used by ingestion and retrieval
- `config.py` - shared chunking and vector-store settings
- `requirements.txt` - Python dependencies
- `results.json` - real sample outputs from three retrieval queries
- `README.md` - setup, run instructions, component justification, and LangGraph node flow

The old EventPulse MERN folders are not required for the Nestack assessment. For a clean submission ZIP, include the assessment files above and exclude `backend/`, `frontend/`, `.venv/`, `vector_store/`, and `node_modules/`.

Before final submission, add your live deployment URL to `README.md` and create the private GitHub repository named `{yourName_Nestack_Submission}` with the required evaluator contributors.
