import os
import time
import json
import streamlit as st
import numpy as np
import faiss
import ollama
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# -----------------------------
# CONFIG
# -----------------------------
DB_FILE = "database/products.json"
VECTOR_INDEX = "vector_store/index.faiss"
DOC_STORE = "vector_store/docs.json"

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "phi3:latest"   # or gemma3:1b

TOP_K_VECTOR = 15
TOP_K_BM25 = 15
TOP_K_FINAL = 10

# -----------------------------
# LISTING QUERY DETECTION
# -----------------------------
LISTING_KEYWORDS = [
    "list", "all", "show", "every", "available", "catalogue",
    "catalog", "how many", "categories", "types", "enumerate",
    "complete", "entire", "full", "summarize", "summary",
    "overview", "what are", "which are", "give me", "tell me all",
    "display", "provide", "count", "total", "each", "every",
    "what models", "which models", "what products", "what parts",
]


def is_listing_query(query):
    q = query.lower()
    return any(kw in q for kw in LISTING_KEYWORDS)


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Uni-Doc-Intel",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>
.main-header {
    font-family: 'Segoe UI';
    border-bottom: 2px solid #0056b3;
    padding-bottom: 10px;
}
.source-box {
    font-size: 0.85em;
    background-color: #1e1e1e;
    color: #e0e0e0;
    padding: 12px;
    border-left: 4px solid #0056b3;
    margin-top: 10px;
    border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.source-title {
    font-weight: bold;
    color: #4fc3f7;
    margin-bottom: 6px;
    display: block;
    font-size: 1.05em;
}
.source-meta {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 4px 12px;
    margin-bottom: 8px;
}
.meta-label {
    color: #9e9e9e;
    font-size: 0.9em;
}
.meta-val {
    color: #fff;
    font-weight: 500;
}
.source-summary {
    color: #b0bec5;
    font-style: italic;
    border-top: 1px dashed #424242;
    padding-top: 6px;
    margin-top: 4px;
}
.query-type-badge {
    font-size: 0.75em;
    padding: 3px 8px;
    border-radius: 12px;
    display: inline-block;
    margin-bottom: 8px;
}
.badge-listing {
    background-color: #e8f5e9;
    color: #2e7d32;
}
.badge-specific {
    background-color: #e3f2fd;
    color: #1565c0;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource
def load_reranker():
    return CrossEncoder(RERANK_MODEL)

embed_model = load_embed_model()
reranker = load_reranker()


# -----------------------------
# ENRICHED CHUNK BUILDER
# -----------------------------
def product_to_chunk(p):
    """Convert a product dict to a richly described text chunk."""
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

    for key, value in p.items():
        if key not in field_labels and value:
            label = key.replace("_", " ").title()
            lines.append(f"{label}: {value}")

    fields_block = "\n".join(lines)

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


# -----------------------------
# BUILD VECTOR DB
# -----------------------------
def build_vector_db():

    with open(DB_FILE) as f:
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

# -----------------------------
# LOAD VECTOR DB
# -----------------------------
@st.cache_resource
def load_vector_db():

    if not os.path.exists(VECTOR_INDEX):
        build_vector_db()

    index = faiss.read_index(VECTOR_INDEX)

    with open(DOC_STORE) as f:
        docs = json.load(f)

    return index, docs

index, docs = load_vector_db()


# -----------------------------
# INIT BM25
# -----------------------------
@st.cache_resource
def init_bm25(_docs):
    tokenized = [doc.lower().split() for doc in _docs]
    return BM25Okapi(tokenized)

bm25 = init_bm25(docs)


# -----------------------------
# HYBRID SEARCH
# -----------------------------
def hybrid_search(query, index, docs):

    # Vector search
    k_vec = min(TOP_K_VECTOR, len(docs))
    q_embedding = embed_model.encode([query])
    D, I = index.search(np.array(q_embedding), k_vec)
    vector_results = [docs[i] for i in I[0] if i < len(docs)]

    # BM25 search
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top = np.argsort(bm25_scores)[::-1][:TOP_K_BM25]
    bm25_results = [docs[i] for i in bm25_top]

    # Merge (deduplicate)
    seen = set()
    merged = []
    for doc in vector_results + bm25_results:
        if doc not in seen:
            seen.add(doc)
            merged.append(doc)

    return merged


# -----------------------------
# RE-RANK
# -----------------------------
def rerank_results(query, candidates):
    if not candidates:
        return []

    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, score in ranked[:TOP_K_FINAL]]


# -----------------------------
# PARSE CHUNK FOR UI
# -----------------------------
def format_source_card(chunk_text, index_num):
    lines = chunk_text.split('\n')
    meta_html = ""
    summary_text = ""
    
    for line in lines:
        if ":" in line and not line.startswith("This is a"):
            parts = line.split(":", 1)
            label = parts[0].strip()
            val = parts[1].strip()
            if label and val:
                meta_html += f'<div><span class="meta-label">{label}:</span> <span class="meta-val">{val}</span></div>'
        elif "This is a" in line or line.strip().endswith("."):
            summary_text += line + " "

    return f"""
    <div class="source-box">
        <span class="source-title">Source [{index_num}]</span>
        <div class="source-meta">
            {meta_html}
        </div>
        <div class="source-summary">{summary_text.strip()}</div>
    </div>
    """

# -----------------------------
# BUILD PROMPT
# -----------------------------
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


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:

    st.title("Uni-Doc-Intel")
    st.caption("Offline Knowledge Assistant")

    if st.button("Clear Chat"):
        st.session_state.messages = []

    st.markdown("---")

    st.write("Knowledge Stats")

    st.metric("Chunks", len(docs))

    if st.button("🔄 Rebuild Knowledge Base"):
        build_vector_db()
        st.cache_resource.clear()
        st.rerun()

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    "<h1 class='main-header'>Uni-Doc-Intel Assistant</h1>",
    unsafe_allow_html=True
)

# -----------------------------
# CHAT HISTORY
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("sources"):

            with st.expander("Sources"):
                for i, s in enumerate(msg["sources"]):
                    st.markdown(format_source_card(s, i+1), unsafe_allow_html=True)

# -----------------------------
# USER QUERY
# -----------------------------
if prompt := st.chat_input("Ask about catalogue parts..."):

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        listing = is_listing_query(prompt)

        # Show query type badge
        if listing:
            st.markdown(
                '<span class="query-type-badge badge-listing">📋 Listing Query</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="query-type-badge badge-specific">🔍 Specific Query</span>',
                unsafe_allow_html=True
            )

        with st.spinner("Searching knowledge..."):

            start = time.time()

            if listing:
                # Pass ALL documents for listing queries
                contexts = docs
            else:
                # Hybrid search + re-rank for specific queries
                candidates = hybrid_search(prompt, index, docs)
                contexts = rerank_results(prompt, candidates)

            latency = time.time() - start

        context_text = "\n\n".join(contexts)

        rag_prompt = build_prompt(prompt, contexts, listing=listing)

        with st.spinner("Generating answer..."):

            response = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role":"user","content":rag_prompt}]
            )

            answer = response["message"]["content"]

        output = f"""{answer}

`Search latency: {latency:.3f}s · Chunks used: {len(contexts)}`
"""

        st.markdown(output)

        with st.expander("Sources"):
            for i, c in enumerate(contexts):
                st.markdown(format_source_card(c, i+1), unsafe_allow_html=True)

        st.session_state.messages.append(
            {
                "role":"assistant",
                "content":output,
                "sources":contexts
            }
        )