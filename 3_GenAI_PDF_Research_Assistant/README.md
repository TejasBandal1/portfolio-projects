# Research Paper PDF Assistant

A local retrieval-augmented generation (RAG) app for asking questions about PDFs. It embeds document chunks, retrieves the most relevant passages, asks a local Ollama model to answer using only those passages, and displays page references so the user can check the answer.

## Public sample data

The included downloader fetches the freely accessible [Attention Is All You Need paper](https://arxiv.org/abs/1706.03762) from arXiv into `data/`. You can also upload your own PDF. The sample document is downloaded locally and is not committed. Respect the terms of documents you upload.

## Run

1. Install [Ollama](https://ollama.com/) and download a local chat model: `ollama pull llama3.2`.
2. Install Python dependencies and download/index the public paper:

```powershell
pip install -r requirements.txt
python download_sample.py
streamlit run app.py
```

3. In the app, select the sample paper or upload a PDF and click **Index PDF**. This creates the local embedding index using `sentence-transformers/all-MiniLM-L6-v2`. Then ask a question. Ollama serves generation locally; no hosted API key is required.

## Approach

Text is split into overlapping chunks, embedded, and ranked by cosine similarity. The top passages are passed to Ollama with instructions to treat PDF text as untrusted evidence, avoid following instructions inside it, admit when the context does not answer, and cite page numbers. Retrieved excerpts are shown below the answer for quick verification.

## Limitations

PDF extraction may miss equations, tables, or scanned pages. Retrieval can miss relevant passages, and a language model can still generate unsupported claims. Always open the cited page and verify. The sample paper corpus is intentionally small; this is a focused document assistant rather than a general knowledge chatbot.
