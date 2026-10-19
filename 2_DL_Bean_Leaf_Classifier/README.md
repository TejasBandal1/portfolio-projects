# Bean Leaf Disease Classifier

A deep learning image classifier for healthy bean leaves, angular leaf spot, and bean rust. It compares a small CNN trained from scratch with a pretrained MobileNetV3-Small fine-tuned by transfer learning. The training script uses the dataset's train/validation/test splits and writes accuracy, per-class precision/recall/F1, macro-F1, a confusion matrix, and model card.

## Dataset

The [Beans dataset](https://huggingface.co/datasets/AI-Lab-Makerere/beans) is a freely available, labeled image dataset hosted on Hugging Face. `datasets` downloads it at first run. Dataset and pretrained-weight terms remain those of their original sources. No images or model weights are checked into the project.

## Run

```powershell
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Training takes longer on CPU. The model artifact and generated reports stay local in `artifacts/` and `reports/`.

## Results

Use `reports/model_comparison.csv`, `reports/metrics.json`, and `reports/confusion_matrix.png` to compare the baseline and transfer-learning model. Scores are created by your run; no results are claimed before evaluation.

## Limitations

The dataset is small and consists of controlled leaf photographs. It does not represent all bean diseases or field conditions. Changes in crop, background, lighting, geography, and camera can reduce performance. This is an educational classifier, not agronomic advice.

## Measured result

With seed 42 and six training epochs, the transfer model improved held-out accuracy from 66.41% to 81.25% and macro-F1 from 0.648 to 0.808 on the dataset's 128-image test split.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| CNN trained from scratch | 66.41% | 0.648 |
| MobileNetV3-Small transfer learning | 81.25% | 0.808 |

The transfer model's bean-rust recall is 58.14%, so it still misses a substantial share of that class. Inspect per-class scores and the confusion matrix in `reports/metrics.json` and `reports/confusion_matrix.png`; the test set is small and these results do not establish field performance.
