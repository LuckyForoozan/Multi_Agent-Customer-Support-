import json
import faiss
from sentence_transformers import SentenceTransformer

# 1. Load the embedding model (only runs once in memory)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Load the saved FAISS index and metadata
index = faiss.read_index("faiss_index.bin")

with open("metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)


def retrieve_context(query: str, top_k: int = 3) -> list[dict]:
    """
    Search the FAISS vector database for the most relevant
    intent/response pairs for a given customer query.
    """
    # Create embedding for the incoming query
    query_vector = embedder.encode([query], normalize_embeddings=True)
    
    # Search top_k nearest matches
    distances, indices = index.search(query_vector, k=top_k)
    
    results = []
    for rank, idx in enumerate(indices[0]):
        if idx != -1:  # valid index
            match = metadata[idx].copy()
            match["similarity_score"] = float(distances[0][rank])
            results.append(match)
            
    return results


# --- Quick Test ---
if __name__ == "__main__":
    test_query = "Can I get a refund for my order?"
    print(f"Testing search for: '{test_query}'\n")
    
    matches = retrieve_context(test_query, top_k=2)
    for i, m in enumerate(matches, 1):
        print(f"--- Result {i} (Score: {m['similarity_score']:.4f}) ---")
        print(f"Intent   : {m['intent']}")
        print(f"Matched  : {m['instruction']}")
        print(f"Response : {m['response'][:120]}...\n")