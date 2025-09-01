# ir-eu-regulations

Turn-key **Step-1 preprocessing** for the EU↔UK Regulatory IR datasets on Hugging Face, producing clean JSONL files you’ll reuse for classical/neural retrieval and evaluation later.

**Highlights**
- POS-aware **NLTK lemmatization**
- **Dynamic stopwords**: add top-N most frequent corpus tokens (configurable)
- Works for both dataset directions: `uk2eu` and `eu2uk`
- Dockerized for repeatability + Colab/Jupyter path for notebooks

---

## Contents

- [What this does](#what-this-does)
- [Dataset variants](#dataset-variants)
- [Project structure](#project-structure)
- [Quick start — Docker (recommended)](#quick-start--docker-recommended)
- [Configuration via `.env`](#configuration-via-env)
- [Run in Google Colab / local Jupyter](#run-in-google-colab--local-jupyter)
- [Script behavior & outputs](#script-behavior--outputs)
- [Troubleshooting](#troubleshooting)
- [License & citation](#license--citation)

---

## What this does

1. Loads `community-datasets/eu_regulatory_ir` with your chosen **config** (`uk2eu` or `eu2uk`).
2. **Cleans** text (lowercase, accents→ASCII, whitespace normalized).
3. **Tokenizes** with `\b[a-z0-9]+\b` (drops 1-char tokens).
4. **Lemmatizes** with NLTK WordNet using POS tags.
5. Builds **dynamic stopwords**: base NLTK English + **top-N** frequent tokens from the **corpus** (N is configurable).
6. Re-filters tokens with the final stopword set.
7. Writes JSONL outputs (corpus + queries) and a `stats.json` summary.

---

## Dataset variants

- **`uk2eu`**: queries = **UK regulations**, corpus = **EU directives**.
- **`eu2uk`**: queries = **EU directives**, corpus = **UK regulations**.

> If you encounter a `KeyError` for a split name, check your `CONFIG` and see [Troubleshooting](#troubleshooting).

---

## Project structure

```
ir-eu-regulations/
├─ src/
│  └─ preprocess.py
├─ data/                        # outputs land here
├─ cache/
│  └─ huggingface/              # speeds up subsequent runs (HF cache)
├─ Dockerfile
├─ docker-compose.yml
└─ .env
```

---

## Quick start — Docker (recommended)

### 0) Prereqs
- Docker Desktop / Engine with Compose v2

### 1) Configure `.env`
Create a `.env` file next to `docker-compose.yml`:

```env
# Dataset configuration (choose one)
CONFIG=uk2eu         # or eu2uk

# Dynamic stopwords: add top-N frequent tokens from the corpus
AUTO_STOPWORDS_TOP=50

# Output directory inside the container (bind-mapped to ./data)
OUT_DIR=/app/data/processed
```

### 2) Build
```bash
docker compose build
```

### 3) Run
```bash
docker compose run --rm preprocess
```

Outputs appear on your host under `./data/processed`:

```
corpus.jsonl
queries_train.jsonl
queries_val.jsonl
queries_test.jsonl
dynamic_stopwords.txt
stats.json
```

### 4) Override on the fly (no YAML edits)

**macOS/Linux**
```bash
CONFIG=eu2uk AUTO_STOPWORDS_TOP=30 OUT_DIR=/app/data/eu2uk_run docker compose run --rm preprocess
```

**Windows PowerShell**
```powershell
$env:CONFIG="eu2uk"
$env:AUTO_STOPWORDS_TOP="30"
$env:OUT_DIR="/app/data/eu2uk_run"
docker compose run --rm preprocess
```

---

## Configuration via `.env`

| Variable              | Description                                                    | Example                 |
|-----------------------|----------------------------------------------------------------|-------------------------|
| `CONFIG`              | Dataset direction                                              | `uk2eu` \| `eu2uk`     |
| `AUTO_STOPWORDS_TOP`  | Add top-N frequent corpus tokens to stopwords                 | `50`                    |
| `OUT_DIR`             | Output path **inside container** (mapped to `./data`)         | `/app/data/processed`   |

You can always override these per run via environment variables (see above).

---

## Run in Google Colab / local Jupyter

**Cell 1 — deps**
```python
!pip -q install datasets nltk tqdm unidecode
import nltk
for pkg in ["stopwords","punkt","wordnet","omw-1.4","averaged_perceptron_tagger_eng"]:
    nltk.download(pkg)
```

**Cell 2 — folders**
```python
import pathlib
root = pathlib.Path("/content/ir-eu-regulations")
(root / "src").mkdir(parents=True, exist_ok=True)
(root / "data").mkdir(parents=True, exist_ok=True)
(root / "cache" / "huggingface").mkdir(parents=True, exist_ok=True)
root
```

**Cell 3 — bring in `preprocess.py` (pick one)**

- Upload from your laptop:
```python
from google.colab import files
import shutil
uploaded = files.upload()  # select your local preprocess.py
shutil.move(next(iter(uploaded.keys())), "/content/ir-eu-regulations/src/preprocess.py")
```

- Or pull from GitHub:
```python
!curl -L "https://raw.githubusercontent.com/<USER>/<REPO>/<BRANCH>/src/preprocess.py"   -o /content/ir-eu-regulations/src/preprocess.py
```

**Cell 4 — run**
```python
!python /content/ir-eu-regulations/src/preprocess.py   --config uk2eu   --auto_stopwords_top 50   --out_dir /content/ir-eu-regulations/data/processed
```

Outputs: `/content/ir-eu-regulations/data/processed`.

---

## Script behavior & outputs

**Behavior**
- Cleans → tokenizes → **lemmatizes** (NLTK WordNet with POS) → builds **dynamic stopwords** from the corpus → re-filters all tokens.
- Supports both dataset configs; ensure your `CONFIG` matches the direction you want.

**Outputs**
- `corpus.jsonl`  
  ```json
  {"document_id":"31977L0794","publication_year":"1977","text":"commission directive ...","tokens":["commission","directive","..."]}
  ```
- `queries_train.jsonl`, `queries_val.jsonl`, `queries_test.jsonl`  
  Each row includes `relevant_documents: [ ... ]`.
- `dynamic_stopwords.txt` — the extra tokens added (for reproducibility)
- `stats.json` — sizes, token lengths, relevance coverage, options, e.g.:
  ```json
  {
    "corpus": {"count": 3930, "avg_tokens": 120.4, "min_tokens": 3, "max_tokens": 8421},
    "train": {"count": 1500, "avg_tokens": 115.2, "min_tokens": 2, "max_tokens": 7110},
    "train_relevance": {"queries": 1500, "avg_rels_per_query": 1.90, "pct_with_at_least_1_rel": 99.1},
    "dynamic_stopwords_added": 50,
    "options": {"config": "uk2eu", "auto_stopwords_top": 50}
  }
  ```

---

## Troubleshooting

- **`KeyError: 'eu_corpus'` or `'uk_corpus'`**  
  You’re likely using the opposite config for that split name. Set `CONFIG=uk2eu` (corpus=`eu_corpus`) or `CONFIG=eu2uk` (corpus=`uk_corpus`).  
  If you still see this, print available splits:
  ```bash
  python - <<'PY'
  from datasets import load_dataset
  ds = load_dataset("community-datasets/eu_regulatory_ir", "uk2eu")  # or eu2uk
  print(list(ds.keys()))
  PY
  ```

- **Missing NLTK data (local/Colab)**  
  Run the downloads in the setup cell:
  ```python
  import nltk
  for p in ["stopwords","punkt","wordnet","omw-1.4","averaged_perceptron_tagger_eng"]:
      nltk.download(p)
  ```

- **Slow first run / repeated downloads**  
  Ensure the HF cache is persisted: in Docker, `./cache/huggingface` is volume-mapped; in Colab, keep the notebook running or mount Drive.

- **Permissions writing to `data/` (Linux)**  
  ```bash
  chmod -R a+rwx data cache
  ```

---

## License & citation

- Code: choose an OSS license (e.g., MIT) for this repository.
- Data: **EU Regulatory IR (EU2UK / UK2EU)** from Hugging Face (`community-datasets/eu_regulatory_ir`).  
  Please consult the dataset card for licensing and citation and cite the authors in any publications.

---

**Questions / improvements?**  
Open an issue or PR with your suggestions (e.g., spaCy lemmatization, section-number joining, passage chunking for neural IR).
