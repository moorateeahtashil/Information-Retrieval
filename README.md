# Text Retrieval Assignment: Classical vs Neural Approaches

This repository implements and evaluates an Information Retrieval (IR) system using both classical and neural approaches. The project compares traditional keyword-based methods (TF-IDF, BM25) with modern neural architectu### Analysis & Insights

### Model Comparison in Financial Domain
1. **Classical Models**
   - TF-IDF provides a solid baseline but struggles with financial terminology variations
   - BM25 shows significant improvement due to better handling of domain-specific terms and document lengths
   - Both models limited in understanding semantic relationships between financial concepts

2. **Neural Models**
   - Bi-encoder effectively captures semantic relationships in financial text
   - Cross-encoder excels at understanding complex financial query-document relationships
   - Two-stage approach provides optimal balance for financial information retrieval
   - Better handling of financial terminology and domain-specific conceptsoder + Cross-encoder) on the FiQA-2018 Financial Domain Question Answering dataset.

## Dataset: FiQA-2018

The project uses the [FiQA-2018 dataset](https://huggingface.co/datasets/mteb/fiqa/viewer/corpus) from the Financial Domain Question Answering challenge. This dataset consists of:

- **Domain**: Financial domain text from StackExchange posts and other financial web sources
- **Content**: Questions and answers about financial topics, including professional financial advice
- **Size**: 
  - Questions: Contains financial domain questions
  - Answers/Documents: Financial expert responses and relevant text passages
- **Source**: Available through Hugging Face datasets

**Key Features**
- Complete implementation of Classical IR models (TF-IDF, BM25)
- Advanced Neural IR pipeline with two-stage architecture
- Comprehensive evaluation metrics (Precision@k, MAP, nDCG)
- FAISS indexing for efficient similarity search
- GPU-accelerated neural models
- Extensive performance comparison and analysisegulations

Turn-key **Step-1 preprocessing** for the EU↔UK Regulatory IR datasets on Hugging Face, producing clean JSONL files you’ll reuse for classical/neural retrieval and evaluation later.

**Highlights**
- POS-aware **NLTK lemmatization**
- **Dynamic stopwords**: add top-N most frequent corpus tokens (configurable)
- Dockerized for repeatability + Colab/Jupyter path for notebooks

---

## System Architecture

### 1. Data Preprocessing
The system processes the FiQA-2018 Financial QA dataset through several stages:
1. Text cleaning (lowercase, ASCII normalization)
2. Financial domain-specific text preprocessing
   - Special handling of financial terms and symbols
   - Preservation of numerical values and currencies
3. Tokenization and lemmatization using NLTK
4. Domain-aware stopword removal
   - Base NLTK stopwords
   - Custom financial domain stopwords
   - Dynamic top-N frequent terms
5. Building inverted index for classical retrieval
6. Preparing corpus and queries for neural models

### 2. Classical IR Implementation
#### TF-IDF Vector Space Model
- Custom implementation from scratch
- Term frequency calculation
- IDF weighting
- Cosine similarity ranking

#### BM25 Model
- Implementation using rank_bm25 library
- Document length normalization
- Term frequency saturation
- Configurable parameters (k1, b)

---

### 3. Neural IR Pipeline

#### First Stage: Bi-encoder Retrieval
- Uses SentenceTransformer (msmarco-distilbert-base-v4)
- Efficient document/query embedding generation
- FAISS indexing for fast similarity search
- Initial candidate retrieval (top-k)

#### Second Stage: Cross-encoder Re-ranking
- Uses cross-encoder/ms-marco-MiniLM-L-6-v2
- Deep pairwise relevance scoring
- Re-ranking of retrieved candidates
- Batch processing with GPU acceleration

## Project Structure

```
Text-Retrieval-Assignment/
├── data/
│   └── processed/          # Preprocessed data files
│       ├── corpus.jsonl    # Processed FiQA corpus
│       ├── queries_*.jsonl # Train/val/test query splits
│       └── stats.json      # Dataset statistics
├── fiqa/                   # Original FiQA dataset
│   ├── corpus.jsonl       # Raw corpus data
│   ├── queries.jsonl      # Raw queries
│   └── processed_data/    # Intermediate processing
├── src/
│   ├── preprocessing.ipynb # Data preparation
│   ├── classical_ir.ipynb  # TF-IDF and BM25
│   └── neural_ir.ipynb    # Bi-encoder + Cross-encoder
├── Dockerfile
└── docker-compose.yml
```

## Performance Results

### Evaluation Metrics
All models are evaluated using:
- Precision@k (k=5,10)
- Mean Average Precision (MAP)
- Normalized DCG (nDCG@10)

### Sample Results
```
Model              P@5    MAP    nDCG@10
----------------------------------------
TF-IDF            0.108  0.354  0.412
BM25              0.132  0.452  0.496
Neural (Bi+Cross) 0.180  0.528  0.563
```

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

## Getting Started

### Requirements
```bash
# Core dependencies
numpy
pandas
torch
transformers
sentence-transformers
faiss-cpu  # or faiss-gpu for GPU support
rank_bm25
nltk
```

### Installation & Setup
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run Docker containers:
```bash
docker compose build
docker compose up
```

### Running the System

1. **Data Preprocessing**
```python
jupyter notebook src/preprocessing.ipynb
```

2. **Classical IR Models**
```python
jupyter notebook src/classical_ir.ipynb
```

3. **Neural IR Models**
```python
jupyter notebook src/neural_ir.ipynb
```

## Analysis & Insights

### Model Comparison
1. **Classical Models**
   - TF-IDF provides a solid baseline but limited to lexical matching
   - BM25 shows significant improvement over TF-IDF due to better length normalization

2. **Neural Models**
   - Bi-encoder enables efficient semantic search
   - Cross-encoder significantly improves ranking quality
   - Two-stage approach balances efficiency and accuracy

### Performance Trade-offs
- **Speed vs Accuracy**: Neural models achieve higher accuracy but require more computational resources
- **Memory Usage**: FAISS indexing optimizes memory usage for large-scale retrieval
- **Preprocessing Impact**: Quality of text preprocessing significantly affects both approaches

## Future Improvements

1. **Financial Domain Specialization**
   - Fine-tune models on larger financial corpora
   - Incorporate financial entity recognition
   - Add support for numerical reasoning and comparison
   - Implement financial domain-specific query expansion

2. **Model Enhancements**
   - Experiment with finance-specialized transformer models
   - Implement financial keyword boosting
   - Add support for temporal financial data

3. **System Optimization**
   - Fine-tune FAISS indexing for financial text characteristics
   - Implement caching for frequent financial queries
   - Optimize batch sizes for better GPU utilization

4. **Evaluation**
   - Add finance-specific evaluation metrics
   - Implement temporal evaluation (considering data freshness)
   - Conduct larger-scale experiments with diverse financial queries
   - Evaluate system performance on different financial sub-domains
    "train_relevance": {"queries": 1500, "avg_rels_per_query": 1.90, "pct_with_at_least_1_rel": 99.1},
    "dynamic_stopwords_added": 50,
    "options": {"config": "uk2eu", "auto_stopwords_top": 50}
  }
  ```

---

## Troubleshooting


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

**Questions / improvements?**  
Open an issue or PR with your suggestions (e.g., spaCy lemmatization, section-number joining, passage chunking for neural IR).
