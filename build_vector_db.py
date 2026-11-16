import os
import json
import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

print("1. Loading Bitext dataset...")
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
data = dataset["train"]

# Extract instructions (queries) and responses
instructions = [item["instruction"] for item in data]
metadata = [
    {
        "intent": item["intent"],
        "category": item["category"],
        "instruction": item["instruction"],
        "response": item["response"]
    }
    for item in data
]

print(f"Loaded {len(instructions)} samples.")

print("2. Loading lightweight embedding model (all-MiniLM-L6-v2)...")
# Small, fast model (~80MB), ideal for low-RAM machines
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("3. Generating embeddings (this may take a few minutes)...")
# Encode in batches to save RAM
embeddings = embedder.encode(
    instructions, 
    batch_size=64, 
    show_progress_bar=True, 
    convert_to_numpy=True,
    normalize_embeddings=True  # For cosine similarity
)

# 4. Create and populate FAISS index
dimension = embeddings.shape[1]  # 384 dimensions
index = faiss.IndexFlatIP(dimension)  # Inner Product on normalized vectors = Cosine Similarity
index.add(embeddings)

print(f"Total vectors indexed: {index.ntotal}")

# 5. Save the index and metadata to disk
print("4. Saving FAISS index and metadata...")
faiss.write_index(index, "faiss_index.bin")

with open("metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("✅ FAISS database and metadata saved successfully!")

# --- Quick Test Search ---
print("\n--- Testing Search ---")
query = "I want to cancel my recent purchase"
query_vector = embedder.encode([query], normalize_embeddings=True)
distances, indices = index.search(query_vector, k=2)

for rank, idx in enumerate(indices[0]):
    match = metadata[idx]
    score = distances[0][rank]
    print(f"\nMatch {rank + 1} (Score: {score:.4f}):")
    print(f"Intent: {match['intent']}")
    print(f"Matched Instruction: {match['instruction']}")
    print(f"Suggested Response: {match['response']}")