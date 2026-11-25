
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("uz_companies")

questions = {
    "A - Factual Questions": [
        "When did Artel open its first factory?",
        "When did Artel release its first smartphone?",
        "What rating did Fitch assign to Artel in 2021?",
        "When was Ucell founded and under what brand name?",
        "How much did Telia sell Ucell for and when was the sale completed?",
        "How many subscribers did Ucell have in December 2018?",
        "What does the name Uzum mean?",
        "Who is the CEO of Uzum?",
        "What was Uzum's revenue in 2025?",
        "What was Uzum's revenue in 2024?",
        "Which investors invested in Uzum in August 2025?",
        "Where is UzAuto Motors headquartered?",
        "What were the previous names of UzAuto Motors?",
        "Who became CEO of UzAuto Motors in August 2021?",
        "When was Uzbekistan Airways founded and when did it begin operations?",
        "What is Uzbekistan Airways' frequent flyer program and how many passengers did it carry in 2024?"
    ],
    "B - Complex Questions": [
        "Which company became Uzbekistan's first tech unicorn?",
        "How many vehicles did UzAuto produce and what was its revenue in 2022?",
        "According to the documents, which company has its headquarters in Asaka, outside Tashkent?",
        "Which foreign companies previously owned Ucell and UzAuto Motors?",
        "How many Boeing 787-8 aircraft does Uzbekistan Airways operate, and how many Boeing 787-9 aircraft are on order?"
    ],
    "C - Unanswerable Questions": [
        "Who founded Artel?",
        "How many employees does Artel have?",
        "What is Ucell's annual revenue?",
        "What is the price of a Chevrolet Cobalt?",
        "What is Payme's valuation?",
        "What is the capital of France?"
    ]
}

SYSTEM_PROMPT = """
You are a RAG assistant answering questions using retrieved documents.

STRICT RULES:
1. Use ONLY facts explicitly stated in the provided context.
2. Do not use outside knowledge or make assumptions.
3. Before answering, check whether the context directly supports the answer.
4. If the answer is missing or not directly supported, respond exactly:
I don't know based on the provided documents.
5. Keep answers concise.
6. Never invent names, numbers, dates, or other facts.
"""

results_file = open("results.md", "w", encoding="utf-8")

results_file.write("# RAG Evaluation Results\n\n")
results_file.write(f"Total documents: 5  \nTotal chunks: {collection.count()}\n\n")

total = 0

for category, question_list in questions.items():
    results_file.write(f"\n## {category}\n\n")

    for question in question_list:
        total += 1
        print(f"\n{total}/27: {question}")

        embedding = model.encode([question]).tolist()

        result = collection.query(
            query_embeddings=embedding,
            n_results=4
        )

        chunks = result["documents"][0]
        metadata = result["metadatas"][0]
        distances = result["distances"][0]

        sources = sorted(set(item["source"] for item in metadata))

        context = ""
        for chunk, meta in zip(chunks, metadata):
            context += f"\nSource: {meta['source']}\n{chunk}\n"

        if not distances or min(distances) > 1.5:
            answer = "I don't know based on the provided documents."
        else:
            response = ollama.chat(
                model="qwen2.5:1.5b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}"
                    }
                ],
                options={"temperature": 0.1}
            )
            answer = response["message"]["content"].strip()

        print("Answer:", answer)

        results_file.write(f"### {total}. {question}\n\n")
        results_file.write(f"**Answer:** {answer}\n\n")
        results_file.write(f"**Sources:** {', '.join(sources)}\n\n")
        results_file.write(f"**Minimum distance:** {min(distances):.4f}\n\n" if distances else "**Minimum distance:** N/A\n\n")

results_file.write("\n---\n")
results_file.write(f"\n**Total questions: {total}**\n")
results_file.close()

print("\nAll tests completed. Results saved to results.md")
