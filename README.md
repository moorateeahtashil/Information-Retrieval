# Text Retrieval Assignment: Classical vs Neural Approaches

This repository implements and evaluates an Information Retrieval (IR) system using both classical and neural approaches. The project compares traditional keyword-based methods (TF-IDF, BM25) with a modern neural architecture (Bi-encoder + Cross-encoder) on the FiQA-2018 Financial Domain Question Answering dataset.

## Dataset: FiQA-2018

The project uses the [FiQA-2018 dataset](https://huggingface.co/datasets/mteb/fiqa/viewer/corpus) from the Financial Domain Question Answering challenge. This dataset consists of:

-   **Domain**: Financial domain text from StackExchange posts and other financial web sources
-   **Content**: Questions and answers about financial topics, including professional financial advice
-   **Size**:
    -   Questions: Contains financial domain questions
    -   Answers/Documents: Financial expert responses and relevant text passages
-   **Source**: Available through Hugging Face datasets

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
│   └── qrels/             # Relevance judgments
├── src/
│   ├── preprocessing.ipynb # Data preparation
│   ├── classical_ir.ipynb  # TF-IDF and BM25
│   ├── neural_ir.ipynb     # Bi-encoder + Cross-encoder
│   └── evaluation.ipynb    # Combined evaluation of all models
├── Dockerfile
└── docker-compose.yml
```

## System Architecture

### 1. Data Preprocessing (`preprocessing.ipynb`)

The system processes the FiQA-2018 dataset through several stages:
1.  **Text Cleaning**: Lowercasing and ASCII normalization.
2.  **Tokenization and Lemmatization**: Using NLTK for part-of-speech aware lemmatization.
3.  **Stopword Removal**: Using a domain-aware stopword list.
4.  **Inverted Index**: Building an inverted index for classical retrieval.

### 2. Classical IR Implementation (`classical_ir.ipynb`)

#### TF-IDF Vector Space Model
-   Custom implementation using `scikit-learn`'s `TfidfVectorizer`.
-   Cosine similarity for ranking.

#### BM25 Model
-   Implementation using the `rank_bm25` library.
-   Includes parameter tuning for `k1` and `b` to optimize performance.

### 3. Neural IR Pipeline (`neural_ir.ipynb`)

#### First Stage: Bi-encoder Retrieval
-   Uses `SentenceTransformer (msmarco-distilbert-base-v4)` for efficient document and query embedding.
-   FAISS indexing for fast similarity search and initial candidate retrieval.

#### Second Stage: Cross-encoder Re-ranking
-   Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` for deep pairwise relevance scoring.
-   Re-ranks the candidates retrieved by the bi-encoder.

## Evaluation (`evaluation.ipynb`)

All models are evaluated using a comprehensive set of metrics:
-   **Precision@k** (k=1, 3, 5, 10, 20)
-   **Recall@k** (k=1, 3, 5, 10, 20)
-   **Mean Average Precision (MAP)**
-   **Mean Reciprocal Rank (MRR)**
-   **Normalized DCG (nDCG@k)** (k=1, 3, 5, 10, 20)
-   **F1@k** (k=1, 3, 5, 10, 20)

The `evaluation.ipynb` notebook provides a combined evaluation of all classical and neural models, including performance and efficiency metrics.

## How to Run

1.  **Preprocessing**: Run the `src/preprocessing.ipynb` notebook to clean and prepare the data.
2.  **Classical IR**: Run the `src/classical_ir.ipynb` notebook to implement and evaluate the TF-IDF and BM25 models.
3.  **Neural IR**: Run the `src/neural_ir.ipynb` notebook to implement and evaluate the neural IR pipeline.
4.  **Combined Evaluation**: Run the `src/evaluation.ipynb` notebook for a comprehensive comparison of all models.