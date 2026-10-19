from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

ROOT = Path(__file__).parent
LABELS = ["angular leaf spot", "bean rust", "healthy"]
WEIGHTS = ROOT / "artifacts" / "mobilenet_plant.pt"

st.set_page_config(page_title="Bean Leaf Classifier", page_icon="🌿")
st.title("Bean Leaf Disease Classifier")
st.caption("Deep learning · MobileNetV3 transfer learning")
if not WEIGHTS.exists():
    st.info("Train the model first with `python train.py`.")
    st.stop()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.mobilenet_v3_small(weights=None)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(LABELS))
model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True)["state_dict"])
model.eval().to(device)
preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
upload = st.file_uploader("Upload a bean leaf image", type=["jpg", "jpeg", "png"])
if upload:
    image = Image.open(upload).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    with torch.no_grad():
        probabilities = model(preprocess(image).unsqueeze(0).to(device)).softmax(1)[0].cpu()
    best = int(probabilities.argmax())
    st.subheader(f"Prediction: {LABELS[best].title()}")
    st.bar_chart({LABELS[i]: float(probabilities[i]) for i in range(len(LABELS))})
    st.caption("Educational model for controlled dataset images; not a diagnosis or crop treatment recommendation.")
