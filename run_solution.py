"""Precision-oriented business entity-resolution pipeline.

Importable (``resolve``) for the web app and runnable as a CLI for the offline
submission:  python run_solution.py [--train-dir ...] [--test-dir ...] [--output-dir ...]
"""
from __future__ import annotations

import argparse
import os
import re
import time
import warnings
from difflib import SequenceMatcher
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from rapidfuzz.distance import Indel

    def ratio(left: str, right: str) -> float:
        return Indel.normalized_similarity(left, right)
except ImportError:  # pragma: no cover - rapidfuzz is optional
    def ratio(left: str, right: str) -> float:
        return SequenceMatcher(None, left, right).ratio()

try:
    from lightgbm import LGBMClassifier
    MODEL_NAME = "LightGBM"
except (ImportError, OSError):  # pragma: no cover - missing package, or missing libgomp on serverless Linux
    LGBMClassifier = None
    MODEL_NAME = "Gradient boosting"
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore", message="X does not have valid feature names")


REQUIRED_COLUMNS = ("entity_id", "business_name", "business_address", "country")
TOP_K = 15
BLOCK_CHUNK = 1024

LEGAL = re.compile(r"\b(private|pvt|limited|ltd|incorporated|inc|corporation|corp|llc|llp|sa|se)\b")
ABBREVIATIONS = {"blr": "bengaluru", "b'lore": "bengaluru", "rd": "road", "st": "street",
                 "hwy": "highway", "pkwy": "parkway", "bldg": "building", "pt": "point",
                 "expwy": "expressway", "hq": "headquarters", "svcs": "services"}


def clean(value):
    text = str(value or "").lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    words = [ABBREVIATIONS.get(word, word) for word in text.split()]
    return " ".join(words)


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["entity_id"] = frame["entity_id"].astype(str)
    for col in ("business_name", "business_address", "country"):
        frame[col] = frame[col].fillna("").map(clean)
    frame["name_clean"] = frame["business_name"].map(lambda x: LEGAL.sub(" ", x)).str.replace(r"\s+", " ", regex=True).str.strip()
    frame["text"] = frame["name_clean"] + " " + frame["business_address"] + " " + frame["country"]
    return frame.reset_index(drop=True)


def token_jaccard(left, right):
    a, b = set(left.split()), set(right.split())
    return len(a & b) / len(a | b) if a | b else 1.0


def pair_features(left, right, cosine: float) -> list[float]:
    values = []
    for column in ("name_clean", "business_address", "text"):
        a, b = left[column], right[column]
        values.extend((ratio(a, b), token_jaccard(a, b)))
    values.append(float(left["country"] == right["country"]))
    values.append(float(cosine))
    return values


def f05(precision, recall):
    return 1.25 * precision * recall / (0.25 * precision + recall) if precision or recall else 0.0


def best_threshold(y_true, scores):
    best = (0.5, -1.0)
    for threshold in np.linspace(0.20, 0.95, 151):
        predicted = scores >= threshold
        tp = np.sum(predicted & (y_true == 1))
        fp = np.sum(predicted & (y_true == 0))
        fn = np.sum(~predicted & (y_true == 1))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        score = f05(precision, recall)
        if score > best[1] or (score == best[1] and threshold > best[0]):
            best = (threshold, score)
    return best


def read_table(path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")


@lru_cache(maxsize=4)
def load_training(train_dir: str):
    source1 = prepare(read_table(os.path.join(train_dir, "train_source1.tsv")))
    targets = prepare(pd.concat([read_table(os.path.join(train_dir, name))
                                 for name in ("train_source2.tsv", "train_source3.tsv")], ignore_index=True))
    truth = read_table(os.path.join(train_dir, "train_ground_truth.tsv")).set_index("source1_entity_id")["matched_entity_ids"]
    return source1, targets, truth.to_dict()


def block(reference_matrix, target_matrix, k: int):
    """TF-IDF blocking: top-k target indices and cosine scores for each reference row."""
    k = min(k, target_matrix.shape[0])
    indices, cosines = [], []
    for start in range(0, reference_matrix.shape[0], BLOCK_CHUNK):
        sims = (reference_matrix[start:start + BLOCK_CHUNK] @ target_matrix.T).toarray()
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k] if k < sims.shape[1] else np.tile(np.arange(sims.shape[1]), (sims.shape[0], 1))
        indices.append(top)
        cosines.append(np.take_along_axis(sims, top, axis=1))
    return np.vstack(indices), np.vstack(cosines)


def resolve(reference: pd.DataFrame, targets: pd.DataFrame, train_dir: str = "dataset/train", top_k: int = TOP_K) -> dict:
    """Match every reference record against the pooled target records.

    Returns the two submission tables plus per-pair scores for display.
    """
    started = time.time()
    if reference.empty or targets.empty:
        raise ValueError("Both the reference file and the comparison files need at least one record.")
    ref, tgt = prepare(reference), prepare(targets)
    train_s1, train_targets, truth = load_training(os.path.abspath(train_dir))

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    vectorizer.fit(pd.concat([train_s1["text"], train_targets["text"], ref["text"], tgt["text"]]))

    # Train on every pair of the (small) labelled training set.
    train_cos = (vectorizer.transform(train_s1["text"]) @ vectorizer.transform(train_targets["text"]).T).toarray()
    x_train, y_train = [], []
    for i, left in train_s1.iterrows():
        positives = set(str(truth.get(left.entity_id, "")).split(","))
        for j, right in train_targets.iterrows():
            x_train.append(pair_features(left, right, train_cos[i, j]))
            y_train.append(int(right.entity_id in positives))
    x_train, y_train = np.asarray(x_train), np.asarray(y_train)
    model = (LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, verbosity=-1, random_state=42)
             if LGBMClassifier else HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=42))
    model.fit(x_train, y_train)
    threshold, train_score = best_threshold(y_train, model.predict_proba(x_train)[:, 1])

    # Candidate blocking, then score only the shortlisted pairs.
    top, cosines = block(vectorizer.transform(ref["text"]), vectorizer.transform(tgt["text"]), top_k)
    ref_records, tgt_records = ref.to_dict("records"), tgt.to_dict("records")
    features = [pair_features(ref_records[i], tgt_records[j], cosines[i, n])
                for i in range(len(ref_records)) for n, j in enumerate(top[i])]
    probabilities = model.predict_proba(np.asarray(features))[:, 1].reshape(top.shape)

    candidate_rows, matching_rows, scored = [], [], {}
    for i, left in enumerate(ref_records):
        ranked = sorted(zip(top[i], probabilities[i]), key=lambda item: (-item[1], item[0]))
        pairs = [(tgt_records[j]["entity_id"], float(p)) for j, p in ranked]
        scored[left["entity_id"]] = pairs
        candidate_rows.append({"source1_entity_id": left["entity_id"], "candidate_entity_ids": ",".join(e for e, _ in pairs)})
        matching_rows.append({"source1_entity_id": left["entity_id"],
                              "matched_entity_ids": ",".join(e for e, p in pairs if p >= threshold)})

    return {
        "candidates": pd.DataFrame(candidate_rows, columns=["source1_entity_id", "candidate_entity_ids"]),
        "matches": pd.DataFrame(matching_rows, columns=["source1_entity_id", "matched_entity_ids"]),
        "scores": scored,
        "threshold": float(threshold),
        "train_f05": float(train_score),
        "model": MODEL_NAME,
        "seconds": time.time() - started,
    }


def main(train_dir="dataset/train", test_dir="dataset/test", output_dir="output"):
    reference = read_table(os.path.join(test_dir, "test_source1.tsv"))
    targets = pd.concat([read_table(os.path.join(test_dir, name)) for name in ("test_source2.tsv", "test_source3.tsv")], ignore_index=True)
    result = resolve(reference, targets, train_dir)
    print(f"Training F_0.5={result['train_f05']:.4f}; selected threshold={result['threshold']:.3f}")
    os.makedirs(output_dir, exist_ok=True)
    result["candidates"].to_csv(os.path.join(output_dir, "candidate_pairs.tsv"), sep="\t", index=False)
    result["matches"].to_csv(os.path.join(output_dir, "matching_results.tsv"), sep="\t", index=False)
    print(f"Wrote {len(reference)} rows in {result['seconds']:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()
    main(args.train_dir, args.test_dir, args.output_dir)
