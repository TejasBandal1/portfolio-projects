from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

ROOT = Path(__file__).parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def load_data() -> pd.DataFrame:
    DATA.mkdir(exist_ok=True)
    source = DATA / "SMSSpamCollection"
    if not source.exists():
        archive = DATA / "sms_spam.zip"
        urllib.request.urlretrieve(URL, archive)
        with zipfile.ZipFile(archive) as zf:
            member = next(n for n in zf.namelist() if n.endswith("SMSSpamCollection"))
            source.write_bytes(zf.read(member))
    records = [line.split("\t", 1) for line in source.read_text(encoding="utf-8").splitlines()
               if "\t" in line]
    df = pd.DataFrame(records, columns=["label", "text"])
    df["label"] = df.label.map({"ham": 0, "spam": 1})
    return df.dropna()


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df.text, df.label, test_size=0.2, random_state=42, stratify=df.label
    )
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                 max_features=80000, sublinear_tf=True)),
    ])
    models = {
        "baseline_nb": Pipeline([("features", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
                                 ("model", MultinomialNB(alpha=0.5))]),
        "final_logistic": Pipeline([("features", features),
                                     ("model", LogisticRegression(max_iter=1500, class_weight="balanced", C=3.0))]),
    }
    result = {"dataset": "UCI SMS Spam Collection", "rows": len(df), "seed": 42,
              "split": "stratified 80/20", "models": {}}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = model.predict_proba(X_test)[:, 1]
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary")
        result["models"][name] = {"accuracy": accuracy_score(y_test, pred),
            "spam_precision": precision, "spam_recall": recall, "spam_f1": f1,
            "roc_auc": roc_auc_score(y_test, score)}
        if name == "final_logistic":
            joblib.dump(model, ARTIFACTS / "spam_pipeline.joblib")
            ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["ham", "spam"], cmap="Blues")
            plt.title("SMS spam detector — held-out confusion matrix")
            plt.tight_layout()
            plt.savefig(REPORTS / "confusion_matrix.png", dpi=160)
            terms = model.named_steps["features"].get_feature_names_out()
            weights = model.named_steps["model"].coef_[0]
            top = sorted(zip(terms, weights), key=lambda pair: abs(pair[1]), reverse=True)[:30]
            pd.DataFrame(top, columns=["term", "logistic_weight"]).to_csv(REPORTS / "top_terms.csv", index=False)
    (REPORTS / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (REPORTS / "model_card.md").write_text(
        "# Model card\n\nEducational SMS spam classifier. Data: UCI SMS Spam Collection. "
        "Evaluation: fixed stratified 80/20 split, seed 42. See metrics.json and confusion_matrix.png. "
        "Known risks: old English SMS domain, dataset bias, false positives, and adversarial messages. "
        "Do not use predictions as a security verdict.\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
