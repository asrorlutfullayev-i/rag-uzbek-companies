
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# 1. Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2. Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="uz_companies")

print(f"\nRAG is ready! Total chunks: {collection.count()}")

# 3. System prompt
SYSTEM_PROMPT = """
You are a helpful assistant answering questions about five Uzbek companies.

Rules:
- Answer ONLY using the provided context.
- Never use your general knowledge.
- Never invent or guess information.
- If the answer is not in the context, respond exactly:
  I don't know based on the provided documents.
- Keep answers short (1-3 sentences unless more detail is requested).
- Always mention the source filename.
"""

# 4. Chat loop
while True:

    question = input("\nYour question: ").strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if not question:
        continue

    # 5. Convert question into embedding
    query_embedding = model.encode([question]).tolist()

    # 6. Retrieve relevant chunks
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=4
    )

    contexts = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print("\nRetrieved sources:")

    for i, (chunk, metadata, distance) in enumerate(
        zip(contexts, metadatas, distances)
    ):
        print(
            f"{i+1}. {metadata['source']} "
            f"(distance: {distance:.4f})"
        )

    # 7. Check retrieval relevance
    if not distances or min(distances) > 1.5:
        print("\nAnswer:")
        print("I don't know based on the provided documents.")
        continue

    # 8. Prepare context
    context_text = ""

    for chunk, metadata in zip(contexts, metadatas):
        context_text += (
            f"\nSource: {metadata['source']}\n"
            f"{chunk}\n"
        )

    # 9. Generate answer using Ollama
    response = ollama.chat(
        model="qwen2.5:1.5b",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"""
Context:
{context_text}

Question:
{question}

Answer using only the provided context.
"""
            }
        ],
        options={
            "temperature": 0.1
        }
    )

    print("\nAnswer:")
    print(response["message"]["content"])
