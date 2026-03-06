import os
import time
import json
import streamlit as st
import numpy as np
import faiss
import ollama
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------
DB_FILE = "database/products.json"
VECTOR_INDEX = "vector_store/index.faiss"
DOC_STORE = "vector_store/docs.json"

EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "phi3:latest"   # or gemma3:1b
TOP_K = 3

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
    background-color: rgba(0,86,179,0.05);
    padding: 10px;
    border-left: 4px solid #0056b3;
    margin-top: 10px;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD EMBEDDING MODEL
# -----------------------------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL)

embed_model = load_embed_model()

# -----------------------------
# BUILD VECTOR DB
# -----------------------------
def build_vector_db():

    with open(DB_FILE) as f:
        data = json.load(f)

    docs = []
    for p in data["products"]:
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
                for s in msg["sources"]:
                    st.markdown(f"""
<div class="source-box">
{s[:250]}...
</div>
""", unsafe_allow_html=True)

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

        with st.spinner("Searching knowledge..."):

            start = time.time()

            q_embedding = embed_model.encode([prompt])

            D, I = index.search(np.array(q_embedding), TOP_K)

            contexts = [docs[i] for i in I[0]]

            latency = time.time() - start

        context_text = "\n\n".join(contexts)

        rag_prompt = f"""
You are an automotive parts assistant.

Use the context below to answer the question.

Context:
{context_text}

Question:
{prompt}
"""

        with st.spinner("Generating answer..."):

            response = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role":"user","content":rag_prompt}]
            )

            answer = response["message"]["content"]

        output = f"""{answer}

`Search latency: {latency:.3f}s`
"""

        st.markdown(output)

        with st.expander("Sources"):
            for c in contexts:
                st.markdown(f"""
<div class="source-box">
{c[:250]}...
</div>
""", unsafe_allow_html=True)

        st.session_state.messages.append(
            {
                "role":"assistant",
                "content":output,
                "sources":contexts
            }
        )