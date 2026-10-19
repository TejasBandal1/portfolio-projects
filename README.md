# Portfolio Projects

Three self-contained projects matching the requested portfolio tracks: one classical machine learning project, one deep learning project, and one generative AI project. Each project folder contains its own README, dataset source, dependencies, scripts, and interactive Streamlit application.

## Projects

1. `1_ML_SMS_Spam_Detector` — UCI SMS Spam Collection; TF-IDF + Naive Bayes baseline compared with Logistic Regression.
2. `2_DL_Bean_Leaf_Classifier` — Hugging Face Beans dataset; small CNN baseline compared with MobileNetV3 transfer learning.
3. `3_GenAI_PDF_Research_Assistant` — public arXiv sample PDF; local RAG, Ollama generation, and page citations.

Install each project's own `requirements.txt` and follow its README. Data, model weights, embeddings, and evaluation outputs are downloaded or generated locally and are excluded from source control.

## Measured model comparisons

- ML: On the held-out UCI split, spam recall rose from 83.89% with Naive Bayes to 95.97% with Logistic Regression.
- DL: On the Beans test split, accuracy rose from 66.41% with a scratch CNN to 81.25% with MobileNetV3 transfer learning.
- GenAI: The PDF assistant retrieves evidence passages and returns cited answers using a local Ollama model. Review the cited pages because generated answers can still be wrong.
