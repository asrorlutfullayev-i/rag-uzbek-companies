# Uzbek Companies RAG

A beginner-friendly local Retrieval-Augmented Generation (RAG) project that answers questions about five Uzbek companies using information from text documents.

This is a learning project. It is **not production-ready**; known weaknesses are listed in the Limitations section.

## Technologies

- Python
- ChromaDB (vector database)
- Sentence Transformers, `all-MiniLM-L6-v2` (embeddings)
- Ollama + Qwen 2.5 1.5B (local LLM)

No API key is required. Everything runs locally.

## Architecture

```text
Text Documents
      |
      v
Document Loading -> Chunking -> Embedding Model -> ChromaDB
                                                      |
User Question -> Query Embedding -> Similarity Search +
                                          |
                                          v
                                  Retrieved Context
                                          |
                                          v
                              Qwen 2.5 via Ollama -> Answer
```

## How It Works

1. `ingest.py` loads the text documents from `data/`.
2. It splits them into smaller chunks and converts each chunk into a vector embedding.
3. Chunks, embeddings and metadata are stored in the ChromaDB collection `uz_companies`.
4. `chat.py` embeds the user's question and retrieves the top-4 most similar chunks.
5. The retrieved context and the question are sent to Qwen 2.5 through Ollama.
6. The model generates an answer based on that context.

RAG does not fine-tune the model: the model stays unchanged, only the context in the prompt changes.

## Project Structure

```text
RAG_Uzbek_Companies/
|-- data/            # 5 source documents (artel, ucell, uzauto_motors, uzbekistan_airways, uzum)
|-- ingest.py        # builds the vector database
|-- chat.py          # interactive question answering
|-- test.py          # runs 27 test questions, writes results.md
|-- debug.py         # retrieval diagnostics
|-- results.md       # latest test results
|-- requirements.txt
|-- README.md
```

`chroma_db/`, `hf_cache/` and `.venv/` are created locally and are not stored in the repository.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed
- Python packages from `requirements.txt` (`chromadb`, `sentence-transformers`, `ollama`)

## Installation

```powershell
git clone https://github.com/asrorlutfullayev-i/rag-uzbek-companies.git
cd rag-uzbek-companies
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ollama pull qwen2.5:1.5b
```

## Run the Project

```powershell
python ingest.py   # build the vector database (run once)
python chat.py     # ask questions; type "exit" to quit
```

## Evaluation

`test.py` runs 27 questions: 16 factual, 5 complex (multi-document) and 6 unanswerable. Results are written to `results.md`. Not all tests pass.

## Limitations

- Retrieval sometimes misses information that exists in the documents.
- The LLM sometimes invents information that is not in the context (hallucination).
- Unanswerable questions do not always get a clear "I don't know" answer.
- The `distance > 1.5` threshold is a rough guess and has not been evaluated.
- Chunk size has not been tuned.
- Complex questions that need several documents are handled poorly.

## Future Improvements

- Tune chunking and retrieval.
- Reduce hallucination and improve abstention.
- Expand the test set and evaluation.
- Add a simple web interface.