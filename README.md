# Text Retrieval: A Comparative Analysis of Classical and Neural Approaches

This project provides an in-depth implementation and evaluation of a sophisticated Information Retrieval (IR) system, contrasting classical keyword-based methods with modern neural architectures. The primary objective is to analyze the trade-offs between traditional models like **TF-IDF** and **BM25** against a state-of-the-art **two-stage neural pipeline** composed of a Bi-Encoder and a Cross-Encoder. All evaluations are performed on the FiQA-2018 dataset, a collection of financial domain questions and answers.

## Dataset: FiQA-2018

The project uses the [FiQA-2018 dataset](https://huggingface.co/datasets/mteb/fiqa/viewer/corpus), a benchmark for financial domain question answering.

-   **Domain**: Financial text from StackExchange posts and other web sources.
-   **Content**: Real-world questions and answers on financial topics, often containing nuanced or specialized language.
-   **Size**: A large collection of questions and expert-written answers, ideal for training and evaluating IR systems.
-   **Source**: Publicly available through Hugging Face Datasets.

## Project Pipeline

The project is structured into a series of Jupyter notebooks, each handling a distinct phase of the IR pipeline:

1.  **`preprocessing.ipynb`**: Data loading, cleaning, and normalization.
2.  **`classical_ir.ipynb`**: Implementation and evaluation of TF-IDF and BM25.
3.  **`neural_ir.ipynb`**: Implementation of the bi-encoder and cross-encoder models.
4.  **`evaluation.ipynb`**: A comprehensive, side-by-side comparison of all models.

## System Architecture

### 1. Data Preprocessing (`preprocessing.ipynb`)

This initial stage is critical for preparing the data for both classical and neural models. The key steps include:

-   **Data Loading**: Ingesting the `corpus.jsonl`, `queries.jsonl`, and `qrels/*.tsv` files.
-   **Text Normalization**:
    -   **Lowercasing**: Converting all text to a uniform case.
    -   **Tokenization**: Breaking down text into individual words or tokens using NLTK.
    -   **Stopword and Punctuation Removal**: Filtering out common, non-informative words and punctuation.
-   **Advanced Linguistic Processing**:
    -   **Part-of-Speech (POS) Tagging**: Identifying the grammatical role of each token (e.g., noun, verb, adjective).
    -   **Lemmatization**: Reducing words to their base or dictionary form, informed by their POS tag (e.g., "running" -> "run").
-   **Inverted Index Construction**: Building an inverted index that maps each term to the documents containing it. This is fundamental for the efficiency of classical models.

### 2. Classical IR Models (`classical_ir.ipynb`)

This notebook explores two of the most well-established techniques in information retrieval.

#### TF-IDF (Term Frequency-Inverse Document Frequency)

-   **Implementation**: A vector space model built using `scikit-learn`'s highly optimized `TfidfVectorizer`.
-   **Ranking**: Documents are ranked based on the **cosine similarity** between their TF-IDF vector and the query vector.

#### BM25 (Okapi BM25)

-   **Implementation**: A probabilistic model implemented using the `rank_bm25` library.
-   **Advantages**: BM25 typically outperforms TF-IDF by incorporating two key concepts:
    -   **Term Frequency Saturation**: It recognizes that the relevance of a term does not grow infinitely with its frequency.
    -   **Document Length Normalization**: It penalizes documents that are longer than average, as they have a higher chance of containing query terms by coincidence.
-   **Fine-Tuning**: The notebook includes a systematic **parameter tuning** process for the `k1` and `b` hyperparameters to optimize the model's performance on the FiQA dataset.

### 3. Neural IR Pipeline (`neural_ir.ipynb`)

This notebook implements a sophisticated two-stage neural network architecture for semantic search.

#### Stage 1: Bi-Encoder for Efficient Retrieval

-   **Model**: Utilizes a pre-trained `SentenceTransformer` model (`msmarco-distilbert-base-v4`), which is fine-tuned for asymmetric search tasks (matching short queries to long documents).
-   **Process**:
    1.  The bi-encoder generates dense vector embeddings for all documents in the corpus.
    2.  These embeddings are indexed using **FAISS (Facebook AI Similarity Search)**, a library for highly efficient vector similarity search.
    3.  When a query is received, it is encoded into a vector, and FAISS is used to rapidly retrieve the top-k most similar document vectors.
-   **Purpose**: This stage acts as a fast "candidate generator," quickly narrowing down the vast corpus to a smaller, more manageable set of potentially relevant documents.

#### Stage 2: Cross-Encoder for Accurate Re-ranking

-   **Model**: Employs a `CrossEncoder` model (`cross-encoder/ms-marco-MiniLM-L-6-v2`), which is specifically designed for re-ranking tasks.
-   **Process**:
    1.  The cross-encoder takes pairs of (query, candidate document) as input.
    2.  It performs a deep, full-attention analysis of the pair, allowing it to capture fine-grained semantic relationships and nuances.
    3.  It outputs a highly accurate relevance score for each pair.
-   **Purpose**: This stage re-ranks the candidates from the bi-encoder, producing a final, more precise list of results. This two-stage approach combines the speed of the bi-encoder with the accuracy of the cross-encoder.

## Comprehensive Evaluation (`evaluation.ipynb`)

The final notebook brings together all the models and evaluates them on a consistent set of metrics, allowing for a direct comparison of their performance and efficiency.

### Performance Metrics

-   **Precision@k, Recall@k, F1@k**: Measure the accuracy of the top-k results.
-   **Mean Average Precision (MAP)**: Provides a single-figure measure of quality across recall levels.
-   **Mean Reciprocal Rank (MRR)**: Measures the rank of the first relevant item.
-   **Normalized Discounted Cumulative Gain (nDCG@k)**: A measure of ranking quality that accounts for the position of relevant items.

### Efficiency Metrics

-   **Query Latency**: The time taken to process a single query.
-   **Throughput**: The number of queries that can be processed per second.
-   **Model and Index Size**: The memory and storage footprint of each model.

## How to Run the Project

1.  **Install Dependencies**: Ensure you have Python 3 and the required libraries installed. You can use the provided `Dockerfile` to create a consistent environment.
2.  **Data Preprocessing**: Execute the `src/preprocessing.ipynb` notebook to prepare the dataset.
3.  **Classical Models**: Run `src/classical_ir.ipynb` to work with the TF-IDF and BM25 models.
4.  **Neural Models**: Run `src/neural_ir.ipynb` to implement the neural pipeline.
5.  **Final Evaluation**: Execute `src/evaluation.ipynb` to see the comprehensive comparison of all models.
