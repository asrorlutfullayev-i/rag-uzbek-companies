\# Uzbek Companies RAG



A beginner-friendly local Retrieval-Augmented Generation (RAG) project that answers questions about five Uzbek companies using information from text documents.



This project demonstrates the basic RAG pipeline: document processing, chunking, embeddings, vector search, and LLM-based answer generation.



\## Technologies



\- Python

\- ChromaDB

\- Sentence Transformers

\- all-MiniLM-L6-v2

\- Ollama

\- Qwen 2.5 1.5B



\## Architecture



Text Documents

&#x20;     |

&#x20;     v

Document Loading

&#x20;     |

&#x20;     v

Text Chunking

&#x20;     |

&#x20;     v

Embedding Model

&#x20;     |

&#x20;     v

ChromaDB

&#x20;     |

&#x20;     v

User Question

&#x20;     |

&#x20;     v

Similarity Search

&#x20;     |

&#x20;     v

Retrieved Context

&#x20;     |

&#x20;     v

Qwen 2.5 via Ollama

&#x20;     |

&#x20;     v

Generated Answer



\## How It Works



1\. Loads five text documents from the `data/` directory.

2\. Splits documents into smaller text chunks.

3\. Converts chunks into vector embeddings.

4\. Stores embeddings and metadata in ChromaDB.

5\. Retrieves relevant chunks for a user question.

6\. Sends the retrieved context and question to Qwen through Ollama.

7\. Generates an answer based on the retrieved information.



\## Project Structure



```text

RAG\_Uzbek\_Companies/

|-- data/

|-- chroma\_db/

|-- ingest.py

|-- chat.py

|-- test.py

|-- debug.py

|-- results.md

|-- requirements.txt

|-- README.md

