#load libraries
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


    # 2. Load embedding model 
model = SentenceTransformer("all-MiniLM-L6-v2")


    # 3. Load resumes (dummy list)
    # In real use: read from files or DB

resumes = [f"Resume text {i}" for i in range(1000)]  # Placeholder

    # 4. Generate embeddings

resume_embeddings = model.encode(resumes, batch_size=32, convert_to_numpy=True)

    # Dimension of embeddings
d = resume_embeddings.shape[1]

    # 5. Build FAISS Index

index = faiss.IndexFlatL2(d)     # L2 similarity index
index.add(resume_embeddings)     # Store 1000 resume vectors

print("FAISS index size:", index.ntotal)

    # 6. JD → Top 5 Resume Search

def search_top_candidates(job_description, top_k=5):
    jd_embedding = model.encode([job_description], convert_to_numpy=True)
    distances, indices = index.search(jd_embedding, top_k)
    return indices[0], distances[0]


    # 7. Run a sample search

job_description = "Looking for a Python developer with ML and API experience."

top_ids, scores = search_top_candidates(job_description)
print("Top candidate indices:", top_ids)
print("Distances:", scores)
