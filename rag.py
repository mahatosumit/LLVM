import json
import os
import faiss
import numpy as np
import ollama

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# -------------------------
# CONFIG
# -------------------------

DATABASE_FILE = "database/products.json"
VECTOR_INDEX = "vector_store/index.faiss"
DOC_STORE = "vector_store/docs.json"

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

LLM_MODEL = "phi3:latest"

TOP_K_VECTOR = 5
TOP_K_BM25 = 5
TOP_K_FINAL = 3

# -------------------------
# LOAD MODELS
# -------------------------

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)

print("Loading re-ranker model...")
reranker = CrossEncoder(RERANK_MODEL)

bm25 = None

# -------------------------
# BUILD VECTOR DATABASE
# -------------------------

def build_vector_db():

    print("Building vector database...")

    with open(DATABASE_FILE, "r") as f:
        data = json.load(f)

    products = data["products"]

    docs = []

    for p in products:

        text = f"""
Category: {p['category']}
Product Code: {p['product_code']}
SAP No: {p['sap_no']}
Description: {p['description']}
Model: {p['model']}
"""

        docs.append(text.strip())

    embeddings = embed_model.encode(docs)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    os.makedirs("vector_store", exist_ok=True)

    faiss.write_index(index, VECTOR_INDEX)

    with open(DOC_STORE, "w") as f:
        json.dump(docs, f)

    print("Vector DB created.")


# -------------------------
# LOAD VECTOR DATABASE
# -------------------------

def load_vector_db():

    index = faiss.read_index(VECTOR_INDEX)

    with open(DOC_STORE, "r") as f:
        docs = json.load(f)

    return index, docs


# -------------------------
# INIT BM25
# -------------------------

def init_bm25(docs):

    global bm25

    tokenized = [doc.lower().split() for doc in docs]

    bm25 = BM25Okapi(tokenized)


# -------------------------
# HYBRID SEARCH
# -------------------------

def hybrid_search(query, index, docs):

    # Vector Search
    query_embedding = embed_model.encode([query])
    D, I = index.search(np.array(query_embedding), TOP_K_VECTOR)

    vector_results = [docs[i] for i in I[0]]

    # BM25 Search
    tokenized_query = query.lower().split()

    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_top = np.argsort(bm25_scores)[::-1][:TOP_K_BM25]

    bm25_results = [docs[i] for i in bm25_top]

    # Merge results
    merged = list(set(vector_results + bm25_results))

    return merged


# -------------------------
# RE-RANK
# -------------------------

def rerank(query, candidates):

    pairs = [[query, doc] for doc in candidates]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_docs = [doc for doc, score in ranked[:TOP_K_FINAL]]

    return top_docs


# -------------------------
# BUILD PROMPT WITH CITATIONS
# -------------------------

def build_prompt(query, contexts):

    context_block = ""

    for i, c in enumerate(contexts):
        context_block += f"[{i+1}] {c}\n\n"

    prompt = f"""
You are an automotive parts assistant.

Use ONLY the provided context to answer the question.

Every factual statement must include a citation
like [1] or [2].

Context:
{context_block}

Question:
{query}

Answer clearly with citations.
"""

    return prompt


# -------------------------
# RAG QUERY
# -------------------------

def ask_question(query, index, docs):

    # Hybrid retrieval
    candidates = hybrid_search(query, index, docs)

    # Re-ranking
    contexts = rerank(query, candidates)

    # Build prompt
    prompt = build_prompt(query, contexts)

    # LLM call
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"], contexts


# -------------------------
# MAIN PROGRAM
# -------------------------

if __name__ == "__main__":

    if not os.path.exists(VECTOR_INDEX):
        build_vector_db()

    index, docs = load_vector_db()

    init_bm25(docs)

    print("\nHybrid RAG system ready.\n")

    while True:

        query = input("Ask: ")

        if query.lower() in ["exit", "quit"]:
            break

        answer, sources = ask_question(query, index, docs)

        print("\nAnswer:\n")
        print(answer)

        print("\nSources:\n")

        for i, s in enumerate(sources):
            print(f"[{i+1}] {s[:200]}...\n")

        print("\n-------------------------\n")