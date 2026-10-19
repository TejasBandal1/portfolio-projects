from pathlib import Path
from urllib.request import urlretrieve

DATA = Path(__file__).parent / "data"
TARGET = DATA / "attention_is_all_you_need.pdf"
URL = "https://arxiv.org/pdf/1706.03762"

if not TARGET.exists():
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"Downloading public arXiv sample to {TARGET.name} ...")
    urlretrieve(URL, TARGET)
print(f"Ready: {TARGET}")
