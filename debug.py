
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("uz_companies")

questions = [
    "How many passengers did Uzbekistan Airways carry in 2024?",
    "Which foreign companies previously owned Ucell and UzAuto Motors?",
    "How many Boeing 787-8 aircraft are in service and how many Boeing 787-9 are on order?",
    "Who founded Artel?",
    "What is Ucell's annual revenue?"
]

for question in questions:
    print("\n" + "=" * 70)
    print("QUESTION:", question)

    embedding = model.encode([question]).tolist()
    result = collection.query(
        query_embeddings=embedding,
        n_results=4
    )

    for i, (doc, meta, distance) in enumerate(zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0]
    )):
        print(f"\n--- CHUNK {i+1} ---")
        print("Source:", meta["source"])
        print("Distance:", round(distance, 4))
        print(doc)
