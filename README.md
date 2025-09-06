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

**Questions / improvements?**  
Open an issue or PR with your suggestions (e.g., spaCy lemmatization, section-number joining, passage chunking for neural IR).
