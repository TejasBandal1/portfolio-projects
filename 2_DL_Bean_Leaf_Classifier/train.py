from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).parent
ARTIFACTS, REPORTS = ROOT / "artifacts", ROOT / "reports"
SEED, EPOCHS, BATCH_SIZE = 42, 6, 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABELS = ["angular_leaf_spot", "bean_rust", "healthy"]


class Leaves(Dataset):
    def __init__(self, split: str, transform):
        self.rows = load_dataset("AI-Lab-Makerere/beans", split=split)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        return self.transform(row["image"].convert("RGB")), int(row["labels"])


def loaders():
    norm = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    train_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12), transforms.ToTensor(), norm])
    eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
    return (
        DataLoader(Leaves("train", train_tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        DataLoader(Leaves("validation", eval_tf), batch_size=BATCH_SIZE, num_workers=0),
        DataLoader(Leaves("test", eval_tf), batch_size=BATCH_SIZE, num_workers=0),
    )


def make_model(transfer: bool):
    if transfer:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(LABELS))
    else:
        model = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(), nn.Dropout(.25), nn.Linear(128, len(LABELS)))
    return model.to(DEVICE)


def train(model, train_loader, val_loader, lr):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_state, best_loss, stale, history = None, float("inf"), 0, []
    for epoch in range(EPOCHS):
        model.train(); train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(); loss = loss_fn(model(images), labels)
            loss.backward(); optimizer.step(); train_loss += loss.item() * len(labels)
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                val_loss += loss_fn(model(images), labels).item() * len(labels)
        val_loss /= len(val_loader.dataset)
        history.append({"epoch": epoch + 1, "train_loss": train_loss / len(train_loader.dataset), "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 2:
            break
    model.load_state_dict(best_state)
    return model, history


def evaluate(model, loader):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            y_true.extend(labels.tolist())
            y_pred.extend(model(images.to(DEVICE)).argmax(1).cpu().tolist())
    return y_true, y_pred


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    ARTIFACTS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    train_loader, val_loader, test_loader = loaders()
    results, histories = {}, {}
    for name, transfer, lr in [("baseline_cnn", False, .001), ("mobilenet_v3_transfer", True, .0008)]:
        model, histories[name] = train(make_model(transfer), train_loader, val_loader, lr)
        y_true, y_pred = evaluate(model, test_loader)
        results[name] = {"accuracy": accuracy_score(y_true, y_pred),
            "classification_report": classification_report(y_true, y_pred, target_names=LABELS,
                output_dict=True, zero_division=0)}
        if transfer:
            torch.save({"state_dict": model.cpu().state_dict(), "labels": LABELS}, ARTIFACTS / "mobilenet_plant.pt")
            ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=LABELS,
                xticks_rotation=25, cmap="Greens")
            plt.tight_layout(); plt.savefig(REPORTS / "confusion_matrix.png", dpi=160); plt.close()
    payload = {"dataset": "Beans", "seed": SEED, "device": DEVICE, "models": results, "history": histories}
    (REPORTS / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame([{"model": name, "accuracy": val["accuracy"],
        "macro_f1": val["classification_report"]["macro avg"]["f1-score"]}
        for name, val in results.items()]).to_csv(REPORTS / "model_comparison.csv", index=False)
    (REPORTS / "model_card.md").write_text("# Model card\n\nEducational bean leaf image classifier. Evaluated on the Beans test split; see metrics.json. Small controlled dataset; not validated for field use or agronomic diagnosis.\n", encoding="utf-8")
    print(json.dumps({name: {"accuracy": val["accuracy"], "macro_f1": val["classification_report"]["macro avg"]["f1-score"]} for name, val in results.items()}, indent=2))


if __name__ == "__main__":
    main()
