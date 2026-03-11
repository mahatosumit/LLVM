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

TOP_K_VECTOR = 10
TOP_K_BM25 = 10
TOP_K_FINAL = 5

# -------------------------
# LISTING / BROAD QUERY DETECTION
# -------------------------

LISTING_KEYWORDS = [
    "list", "all", "show", "every", "available", "catalogue",
    "catalog", "how many", "categories", "types", "enumerate",
    "complete", "entire", "full", "summarize", "summary",
    "overview", "what are", "which are", "give me", "tell me all",
    "display", "provide", "count", "total", "each", "every",
    "what models", "which models", "what products", "what parts",
]


def is_listing_query(query):
    """Detect if the query requires listing/aggregation over the whole knowledge base."""
    q = query.lower()
    return any(kw in q for kw in LISTING_KEYWORDS)


# -------------------------
# LOAD MODELS
# -------------------------

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)

print("Loading re-ranker model...")
reranker = CrossEncoder(RERANK_MODEL)

bm25 = None

# -------------------------
# BUILD ENRICHED TEXT CHUNK FROM A PRODUCT ENTRY
# -------------------------

def product_to_chunk(p):
    """
    Convert a product dict to a richly described text chunk.
    Handles ANY fields present in the JSON — future-proof for
    data like ref_no, qty, remarks, assembly, figure, etc.
    """

    # Build explicit field lines
    lines = []
    field_labels = {
        "category": "Category",
        "product_code": "Product Code",
        "sap_no": "SAP Number",
        "description": "Description",
        "model": "Vehicle Model",
        "ref_no": "Reference Number",
        "part_no": "Part Number",
        "qty": "Quantity",
        "remarks": "Remarks",
        "assembly": "Assembly",
        "figure": "Figure",
    }

    for key, label in field_labels.items():
        if key in p and p[key]:
            lines.append(f"{label}: {p[key]}")

    # Also include any extra fields not in the map above
    for key, value in p.items():
        if key not in field_labels and value:
            label = key.replace("_", " ").title()
            lines.append(f"{label}: {value}")

    fields_block = "\n".join(lines)

    # Build a natural-language summary sentence for better semantic matching
    desc = p.get("description", p.get("category", "Part"))
    model = p.get("model", "")
    category = p.get("category", "")
    sap = p.get("sap_no", p.get("part_no", ""))
    code = p.get("product_code", "")

    summary_parts = [f"This is a {desc}"]
    if model:
        summary_parts.append(f"for the {model}")
    if category:
        summary_parts.append(f"in the {category} category")
    summary = " ".join(summary_parts) + "."

    if sap:
        summary += f" The SAP/Part number is {sap}."
    if code:
        summary += f" The product code is {code}."

    return f"{fields_block}\n{summary}"


# -------------------------
# BUILD VECTOR DATABASE
# -------------------------

def build_vector_db():

    print("Building vector database...")

    with open(DATABASE_FILE, "r") as f:
        data = json.load(f)

    products = data.get("products", data.get("parts", []))

    docs = []

    for p in products:
        chunk = product_to_chunk(p)
        docs.append(chunk)

    embeddings = embed_model.encode(docs)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    os.makedirs("vector_store", exist_ok=True)

    faiss.write_index(index, VECTOR_INDEX)

    with open(DOC_STORE, "w") as f:
        json.dump(docs, f)

    print(f"Vector DB created with {len(docs)} chunks.")


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
    k_vec = min(TOP_K_VECTOR, len(docs))
    query_embedding = embed_model.encode([query])
    D, I = index.search(np.array(query_embedding), k_vec)

    vector_results = [docs[i] for i in I[0] if i < len(docs)]

    # BM25 Search
    tokenized_query = query.lower().split()

    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_top = np.argsort(bm25_scores)[::-1][:TOP_K_BM25]

    bm25_results = [docs[i] for i in bm25_top]

    # Merge results (deduplicate)
    seen = set()
    merged = []
    for doc in vector_results + bm25_results:
        if doc not in seen:
            seen.add(doc)
            merged.append(doc)

    return merged


# -------------------------
# RE-RANK
# -------------------------

def rerank(query, candidates):

    if not candidates:
        return []

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

def build_prompt(query, contexts, listing=False):

    context_block = ""

    for i, c in enumerate(contexts):
        context_block += f"[{i+1}] {c}\n\n"

    if listing:
        instruction = (
            "The user is asking for a list, summary, or count.\n"
            "You MUST include EVERY relevant item from the context below.\n"
            "Present your answer STRICTLY as a Markdown table.\n"
            "Do NOT write a paragraph or a comma-separated list.\n"
            "Include columns for all important details like Category, SAP Number, Product Code, and Model.\n"
            "Do NOT skip any item. Do NOT summarize or abbreviate."
        )
    else:
        instruction = (
            "Answer the question precisely using only the context below.\n"
            "If the answer is not in the context, say: "
            "'I don't have this information in my knowledge base.'"
        )

    prompt = f"""You are an automotive parts catalogue assistant.

STRICT RULES:
- Use ONLY the provided context to answer. Do NOT use any prior knowledge.
- Do NOT fabricate, guess, or infer information that is not explicitly in the context.
- If the information is not in the context, say: "I don't have this information in my knowledge base."
- Every factual statement must include a citation like [1] or [2].

{instruction}

Context:
{context_block}

Question:
{query}

Answer:"""

    return prompt


# -------------------------
# RAG QUERY
# -------------------------

def ask_question(query, index, docs):

    listing = is_listing_query(query)

    if listing:
        # For listing queries, pass ALL documents as context
        contexts = docs
    else:
        # Hybrid retrieval + re-ranking for specific queries
        candidates = hybrid_search(query, index, docs)
        contexts = rerank(query, candidates)

    # Build prompt
    prompt = build_prompt(query, contexts, listing=listing)

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

    print(f"\nHybrid RAG system ready. ({len(docs)} knowledge chunks loaded)\n")

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