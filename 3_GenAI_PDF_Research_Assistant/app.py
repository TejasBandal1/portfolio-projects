from pathlib import Path
import streamlit as st
from rag import INDEX, build_index, generate, retrieve

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

st.set_page_config(page_title="Research Paper Assistant", page_icon="📚")
st.title("Research Paper PDF Assistant")
st.caption("GenAI · Local RAG with page citations")
st.write("Index the public sample paper or upload a PDF. Ollama must be running locally.")
sample = DATA / "attention_is_all_you_need.pdf"
options = ["Upload a PDF"] + (["Use arXiv sample: Attention Is All You Need"] if sample.exists() else [])
choice = st.radio("Document", options, horizontal=True)
uploaded = st.file_uploader("Upload PDF", type=["pdf"]) if choice == "Upload a PDF" else None
model_name = st.text_input("Ollama model", value="llama3.2")

if st.button("Index PDF", type="primary"):
    if uploaded:
        target = DATA / uploaded.name
        target.write_bytes(uploaded.getvalue())
    elif choice.startswith("Use arXiv"):
        target = sample
    else:
        target = None
    if target:
        try:
            with st.spinner("Extracting pages and building embeddings..."):
                count = build_index(target)
            st.session_state["indexed"] = True
            st.success(f"Indexed {count} passages from {target.name}.")
        except Exception as exc:
            st.error(str(exc))

if INDEX.exists() or st.session_state.get("indexed"):
    question = st.text_input("Ask a question about the indexed PDF")
    if st.button("Ask") and question.strip():
        try:
            with st.spinner("Retrieving evidence and generating an answer..."):
                passages = retrieve(question)
                answer = generate(question, passages, model=model_name)
            st.markdown("### Answer")
            st.write(answer)
            st.markdown("### Retrieved passages")
            for item in passages:
                with st.expander(f"Page {item['page']} · relevance {item['score']:.2f}"):
                    st.write(item["text"])
        except Exception as exc:
            st.error(str(exc))
else:
    st.info("Index a PDF to start asking questions.")
