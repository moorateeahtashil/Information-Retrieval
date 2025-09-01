#!/usr/bin/env python3
# Step 1: Preprocess EURegIR and build an inverted index.
# This version is optimized for performance and includes indexing logic.

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from datasets import load_dataset, Dataset
from nltk.corpus import stopwords as nltk_sw
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from tqdm import tqdm
from unidecode import unidecode

# --- Constants ---
DATASET_NAME = "community-datasets/eu_regulatory_ir"
CORPUS_SPLIT = "uk_corpus"
QUERY_SPLITS = ("train", "validation", "test")
WORD_RE = re.compile(r"\b[a-z0-9]+\b", flags=re.IGNORECASE)

# --- Preprocessor Class for Cleaner State Management ---

class Preprocessor:
    """Encapsulates all text processing logic and state."""
    def __init__(self, use_pos_tagging: bool = False):
        self.wnl = WordNetLemmatizer()
        self.use_pos_tagging = use_pos_tagging
        self.stopset = set()
        print(f"Preprocessor initialized. Using POS tagging: {self.use_pos_tagging}")

    @staticmethod
    def clean_text(text: str) -> str:
        """Basic text cleaning."""
        if not text:
            return ""
        text = unidecode(text)
        text = text.replace("\u00a0", " ")
        text = re.sub(r"\s+", " ", text)
        return text.lower().strip()

    def _lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatizes tokens, with or without POS tagging."""
        if not self.use_pos_tagging:
            return [self.wnl.lemmatize(t) for t in tokens]
        
        tagged = pos_tag(tokens, lang='eng')
        return [self.wnl.lemmatize(t, self._map_pos(p)) for t, p in tagged]

    @staticmethod
    def _map_pos(tag: str) -> str:
        """Maps NLTK POS tags to WordNet tags."""
        if tag.startswith("J"): return wn.ADJ
        if tag.startswith("V"): return wn.VERB
        if tag.startswith("N"): return wn.NOUN
        if tag.startswith("R"): return wn.ADV
        return wn.NOUN

    def tokenize_and_lemma(self, text: str) -> List[str]:
        """Full pipeline for turning raw text into lemmatized tokens (before stopwords)."""
        cleaned_text = self.clean_text(text)
        tokens = [t for t in WORD_RE.findall(cleaned_text) if len(t) > 1]
        return self._lemmatize_tokens(tokens)
        
    def process_document(self, example: Dict) -> Dict:
        """Processes a single document/query dictionary for mapping."""
        text = example.get("text", "")
        tokens = self.tokenize_and_lemma(text)
        example['tokens'] = [t for t in tokens if t not in self.stopset]
        example['text'] = self.clean_text(text)
        return example

    def build_dynamic_stopset(self, corpus: Dataset, top_n: int):
        """Builds the stopword set from base NLTK and top N corpus tokens."""
        print("Building stopword set...")
        base_stop = set(nltk_sw.words("english"))
        
        if top_n <= 0:
            self.stopset = base_stop
            print(f"Using {len(self.stopset)} base stopwords.")
            return [], base_stop

        print(f"Calculating token frequencies from corpus to find top {top_n} dynamic stopwords...")
        freq = Counter()
        for text in tqdm(corpus['text'], desc="Corpus Freq Pass"):
            tokens = self.tokenize_and_lemma(text)
            freq.update(tokens)
            
        dyn_candidates = [t for t, _ in freq.most_common() if t not in base_stop]
        dynamic_topN = set(dyn_candidates[:top_n])
        
        self.stopset = base_stop.union(dynamic_topN)
        print(f"Stopset built: {len(base_stop)} base + {len(dynamic_topN)} dynamic = {len(self.stopset)} total.")
        return sorted(list(dynamic_topN))

# --- IO and Stats Helpers ---
def write_jsonl(path: Path, dataset: Dataset, columns: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in tqdm(dataset, desc=f"Writing {path.name}"):
            row_dict = {col: r.get(col) for col in columns}
            f.write(json.dumps(row_dict, ensure_ascii=False))
            f.write("\n")

def get_stats(dataset: Dataset, is_query: bool = False) -> Dict:
    stats = {}
    lens = [len(r["tokens"]) for r in dataset]
    if not lens:
        stats.update({"count": 0, "avg_tokens": 0.0, "min_tokens": 0, "max_tokens": 0})
    else:
        stats.update({"count": len(lens), "avg_tokens": round(sum(lens) / len(lens), 2), "min_tokens": min(lens), "max_tokens": max(lens)})
    if is_query:
        total_rels = sum(len(r.get("relevant_documents") or []) for r in dataset)
        with_rel = sum(1 for r in dataset if len(r.get("relevant_documents") or []) > 0)
        n = len(dataset)
        stats["relevance"] = {"queries": n, "avg_rels_per_query": round(total_rels / n, 2) if n > 0 else 0, "pct_with_at_least_1_rel": round(100.0 * with_rel / n, 1) if n > 0 else 0}
    return stats


# --- Indexing Function ---
def build_inverted_index(processed_corpus: Dataset) -> Dict[str, List[str]]:
    """
    Builds a simple inverted index from the preprocessed corpus.
    The index maps a token to a sorted list of document IDs that contain it.
    """
    print("\nBuilding inverted index...")
    inverted_index = defaultdict(set)
    
    for doc in tqdm(processed_corpus, desc="Indexing"):
        doc_id = doc["document_id"]
        tokens = doc["tokens"]
        for token in tokens:
            inverted_index[token].add(doc_id)
            
    print("Finalizing index (sorting posting lists)...")
    final_index = {token: sorted(list(doc_ids)) for token, doc_ids in inverted_index.items()}
    
    return final_index

def parse_args():
    default_config = os.environ.get("CONFIG", "eu2uk")
    default_out_dir = os.environ.get("OUT_DIR", "data/processed")
    default_top = int(os.environ.get("AUTO_STOPWORDS_TOP", "50"))
    num_procs = os.cpu_count()
    ap = argparse.ArgumentParser(description="Optimized preprocessing of EURegIR with NLTK.")
    ap.add_argument("--config", type=str, default=default_config, help="Dataset config (uk2eu or eu2uk)")
    ap.add_argument("--out_dir", type=str, default=default_out_dir, help="Output directory")
    ap.add_argument("--auto_stopwords_top", type=int, default=default_top, help="Add top-N frequent corpus tokens")
    ap.add_argument("--num_proc", type=int, default=num_procs, help="Number of CPU cores to use for processing")
    ap.add_argument("--use_pos_tagging", action="store_true", help="Use slower, POS-based lemmatization. Default is faster.")
    return ap.parse_args()


# --- Main Execution ---
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Initialize preprocessor and load data
    preprocessor = Preprocessor(use_pos_tagging=args.use_pos_tagging)
    print(f"Loading dataset: {DATASET_NAME}/{args.config}")
    ds = load_dataset(DATASET_NAME, args.config)

    # 2. Build the combined stopword list
    dynamic_stopwords = preprocessor.build_dynamic_stopset(ds[CORPUS_SPLIT], args.auto_stopwords_top)

    # 3. Process all splits in parallel
    print(f"\nProcessing all datasets using {args.num_proc} cores...")
    processing_fn = preprocessor.process_document
    corpus_processed = ds[CORPUS_SPLIT].map(processing_fn, num_proc=args.num_proc)
    train_processed = ds["train"].map(processing_fn, num_proc=args.num_proc)
    val_processed = ds["validation"].map(processing_fn, num_proc=args.num_proc)
    test_processed = ds["test"].map(processing_fn, num_proc=args.num_proc)

    # 4. Build and save the inverted index
    inverted_index = build_inverted_index(corpus_processed)
    index_path = out_dir / "inverted_index.json"
    print(f"Saving inverted index to {index_path}...")
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(inverted_index, f, indent=2)

    # 5. Write processed data to disk
    print("\nWriting processed files...")
    write_jsonl(out_dir / "corpus.jsonl", corpus_processed, ["document_id", "publication_year", "text", "tokens"])
    query_cols = ["document_id", "publication_year", "text", "tokens", "relevant_documents"]
    write_jsonl(out_dir / "queries_train.jsonl", train_processed, query_cols)
    write_jsonl(out_dir / "queries_val.jsonl", val_processed, query_cols)
    write_jsonl(out_dir / "queries_test.jsonl", test_processed, query_cols)

    # 6. Generate and write stats
    stats = {
        "corpus": get_stats(corpus_processed),
        "train": get_stats(train_processed, is_query=True),
        "validation": get_stats(val_processed, is_query=True),
        "test": get_stats(test_processed, is_query=True),
        "dynamic_stopwords_added": len(dynamic_stopwords),
        "index_stats": {
            "unique_terms": len(inverted_index)
        },
        "options": vars(args)
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out_dir / "dynamic_stopwords.txt").write_text("\n".join(dynamic_stopwords), encoding="utf-8")

    print("\n=== Done ===")
    print(f"Wrote files to: {out_dir.resolve()}")
    print(f"  - Inverted index saved to: {index_path.name}")
    print("\nSummary:")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    try:
        _ = nltk_sw.words("english")
    except LookupError:
        raise RuntimeError(
            "Missing NLTK data. Please run this command first:\n"
            "python -c \"import nltk; nltk.download('stopwords'); nltk.download('punkt'); "
            "nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('averaged_perceptron_tagger');\""
        )
    main()