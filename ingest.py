
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load documents
documents = []

for file in sorted(Path("data").glob("*.txt")):
    documents.append({
        "text": file.read_text(encoding="utf-8"),
        "source": file.name
    })

print("Loaded documents:", len(documents))

# Better chunking
chunks = []

for doc in documents:
    lines = [line.strip() for line in doc["text"].splitlines() if line.strip()]

    current_chunk = ""
    chunk_index = 0

    for line in lines:
        if len(current_chunk) + len(line) > 700 and current_chunk:
            chunks.append({
                "text": current_chunk,
                "source": doc["source"],
                "chunk_index": chunk_index
            })
            chunk_index += 1

            current_chunk = current_chunk[-100:] + "\n" + line
        else:
            current_chunk += "\n" + line

    if current_chunk.strip():
        chunks.append({
            "text": current_chunk,
            "source": doc["source"],
            "chunk_index": chunk_index
        })

print("Created chunks:", len(chunks))

# Rebuild database
client = chromadb.PersistentClient(path="chroma_db")

try:
    client.delete_collection("uz_companies")
except Exception:
    pass

collection = client.create_collection("uz_companies")

# Save embeddings
for i in range(0, len(chunks), 32):
    batch = chunks[i:i + 32]

    texts = [x["text"] for x in batch]
    embeddings = model.encode(texts).tolist()

    ids = [
        f'{x["source"]}_{x["chunk_index"]}'
        for x in batch
    ]

    metadatas = [
        {
            "source": x["source"],
            "chunk_index": x["chunk_index"]
        }
        for x in batch
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

print("Ingestion completed!")
print("Total chunks:", collection.count())
