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
LLM_MODEL = "gemma4:latest"

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
    page_title="DOC-INTEL Engine",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>
/* Global Styling */
[data-testid="stAppViewContainer"] {
    background-color: #0f172a;
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background-color: #1e293b;
    border-right: 1px solid #334155;
    z-index: 100;
}
/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Industrial UI Elements */
.main-header {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background: linear-gradient(90deg, #0ea5e9 0%, #2563eb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #334155;
    margin-bottom: 2rem;
}

.system-badge {
    background-color: #0369a1;
    color: #f0f9ff;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.7rem;
    font-weight: 600;
    display: inline-block;
    margin-top: -10px;
    margin-bottom: 20px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.source-box {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-left: 4px solid #38bdf8;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    transition: transform 0.2s;
}
.source-box:hover {
    transform: translateX(5px);
    border-color: #38bdf8;
}
.source-title {
    color: #38bdf8;
    font-weight: 700;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
}
.source-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 1.5rem;
    margin-bottom: 0.5rem;
    font-size: 0.8rem;
}
.meta-label { color: #94a3b8; }
.meta-val { color: #f1f5f9; font-weight: 500; }
.source-summary {
    color: #cbd5e1;
    font-size: 0.85rem;
    line-height: 1.5;
    padding-top: 0.5rem;
    border-top: 1px solid #334155;
}

/* Chat Overrides */
[data-testid="stChatMessage"] {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 0.75rem !important;
}

.query-type-badge {
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
    text-transform: uppercase;
}
.badge-listing { background-color: #064e3b; color: #34d399; }
.badge-specific { background-color: #1e3a8a; color: #93c5fd; }
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


def reciprocal_rank_fusion(list_of_ranked_lists, k=60):
    from collections import defaultdict
    rrf_scores = defaultdict(float)
    for ranked_list in list_of_ranked_lists:
        for rank, doc in enumerate(ranked_list):
            rrf_scores[doc] += 1.0 / (k + rank + 1)
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in sorted_docs]

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

    # RRF Fusion
    merged = reciprocal_rank_fusion([vector_results, bm25_results])

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
        elif "This is a" in line or (len(line.strip()) > 20 and line.strip().endswith(".")):
            summary_text += line + " "

    return f"""
    <div class="source-box">
        <div class="source-title">REFERENCE NODE [{index_num}]</div>
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
        context_block += f"[[{i+1}]] {c}\n\n"

    if listing:
        instruction = (
            "OBJECTIVE: Provide a comprehensive inventory/list extraction.\n"
            "REQUIREMENTS:\n"
            "1. Identify EVERY unique part/item mentioned in the context.\n"
            "2. Output strictly as a Markdown table (Columns: Category, Model, Part No, Description, Qty).\n"
            "3. Do not omit any technical specifications.\n"
            "4. Do not provide conversational filler."
        )
    else:
        instruction = (
            "OBJECTIVE: Provide precise technical information based on query.\n"
            "REQUIREMENTS:\n"
            "1. Answer strictly using the provided reference nodes.\n"
            "2. If multiple models match, specify each separately.\n"
            "3. If the data is absent, state: 'N/A - Information not in knowledge base.'\n"
            "4. Match citations to reference node IDs (e.g., [[1]])."
        )

    prompt = f"""### SYSTEM: DOC-INTEL INDUSTRIAL ASSISTANT
ROLE: Expert Automotive Parts & Catalog Intelligence Engine.
RULES:
- RESPONSE MUST BE FULLY OFFLINE KNOWLEDGE BASED.
- MANDATORY CITATIONS: Every factual unit must be followed by its source ID [X].
- ACCURACY: 100% precision. Do not hypothesize or fabricate data.

{instruction}

### REFERENCE KNOWLEDGE:
{context_block}

### USER QUERY:
{query}

### ASSISTANT RESPONSE:"""

    return prompt


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/000000/settings.png", width=60)
    st.title("DOC-INTEL Engine")
    st.markdown('<div class="system-badge">Offline Machine Intelligence</div>', unsafe_allow_html=True)
    
    st.markdown("### 👁️ Vision Input")
    uploaded_image = st.file_uploader("Scan Part Image...", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        st.image(uploaded_image, caption="Current Scan", use_column_width=True)

    st.markdown("---")
    if st.button("Clear History"):
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
    "<h1 class='main-header'>DOC-INTEL-ASSISTANT</h1>",
    unsafe_allow_html=True
)
st.markdown('<div class="system-badge">Industrial Offline Intelligence System</div>', unsafe_allow_html=True)

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
            messages = [{"role": "user", "content": rag_prompt}]
            
            # 👁️ Vision logic (Gemma 4 native support)
            if uploaded_image:
                image_bytes = uploaded_image.getvalue()
                messages[0]["images"] = [image_bytes]

            try:
                response = ollama.chat(
                    model=LLM_MODEL,
                    messages=messages
                )
                answer = response["message"]["content"]
            except Exception as e:
                if "not found" in str(e).lower():
                    answer = f"⚠️ **Error:** The model `{LLM_MODEL}` was not found in Ollama.\n\n" \
                             f"Please run `ollama pull {LLM_MODEL}` in your terminal."
                else:
                    answer = f"⚠️ **Ollama Error:** {str(e)}"

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
