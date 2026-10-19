from pathlib import Path
import joblib
import streamlit as st

MODEL = Path(__file__).parent / "artifacts" / "spam_pipeline.joblib"
st.set_page_config(page_title="SMS Spam Detector", page_icon="📨")
st.title("SMS Spam Detector")
st.caption("Classical ML · TF-IDF + Logistic Regression")
if not MODEL.exists():
    st.info("Train the model first: `python train.py`")
    st.stop()
pipeline = joblib.load(MODEL)
message = st.text_area("Enter a message", placeholder="Congratulations! You have won...")
if st.button("Analyze message", type="primary") and message.strip():
    spam_probability = float(pipeline.predict_proba([message])[0, 1])
    st.metric("Spam score", f"{spam_probability:.1%}")
    st.success("Likely spam" if spam_probability >= 0.5 else "Likely ham")
    st.caption("A model score is not proof. Avoid opening links in messages you do not trust.")
