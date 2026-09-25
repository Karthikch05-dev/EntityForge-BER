"""Train and run a precision-oriented entity-resolution pipeline."""
import argparse
import os
import re
import time
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - useful on minimal installations
    LGBMClassifier = None
from sklearn.ensemble import HistGradientBoostingClassifier


LEGAL = re.compile(r"\b(private|pvt|limited|ltd|incorporated|inc|corporation|corp|llc|llp|sa|se)\b")
ABBREVIATIONS = {"blr": "bengaluru", "b'lore": "bengaluru", "rd": "road", "st": "street",
                 "hwy": "highway", "pkwy": "parkway", "bldg": "building", "pt": "point",
                 "expwy": "expressway", "hq": "headquarters", "svcs": "services"}


def clean(value):
    text = str(value or "").lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    words = [ABBREVIATIONS.get(word, word) for word in text.split()]
    return " ".join(words)


def prepare(frame):
    frame = frame.copy()
    for col in ("business_name", "business_address", "country"):
        frame[col] = frame[col].fillna("").map(clean)
    frame["name_clean"] = frame["business_name"].map(lambda x: LEGAL.sub(" ", x)).str.replace(r"\s+", " ", regex=True).str.strip()
    frame["text"] = frame["name_clean"] + " " + frame["business_address"] + " " + frame["country"]
    return frame


def token_jaccard(left, right):
    a, b = set(left.split()), set(right.split())
    return len(a & b) / len(a | b) if a | b else 1.0


def pair_features(left, right, vectorizer):
    values = []
    for column in ("name_clean", "business_address", "text"):
        a, b = left[column], right[column]
        values.extend((SequenceMatcher(None, a, b).ratio(), token_jaccard(a, b)))
    values.append(float(left["country"] == right["country"]))
    values.append(float(cosine_similarity(vectorizer.transform([left["text"]]), vectorizer.transform([right["text"]]))[0, 0]))
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


def load(path):
    return prepare(pd.read_csv(path, sep="\t", dtype=str))


def main(train_dir="dataset/train", test_dir="dataset/test", output_dir="output"):
    started = time.time()
    train_s1 = load(os.path.join(train_dir, "train_source1.tsv"))
    train_targets = pd.concat([load(os.path.join(train_dir, name)) for name in ("train_source2.tsv", "train_source3.tsv")], ignore_index=True)
    test_s1 = load(os.path.join(test_dir, "test_source1.tsv"))
    test_targets = pd.concat([load(os.path.join(test_dir, name)) for name in ("test_source2.tsv", "test_source3.tsv")], ignore_index=True)

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    vectorizer.fit(pd.concat([train_s1["text"], train_targets["text"], test_s1["text"], test_targets["text"]]))

    truth = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t", dtype=str).set_index("source1_entity_id")
    x_train, y_train = [], []
    for _, left in train_s1.iterrows():
        positives = set(str(truth.loc[left.entity_id, "matched_entity_ids"]).split(","))
        for _, right in train_targets.iterrows():
            x_train.append(pair_features(left, right, vectorizer))
            y_train.append(int(right.entity_id in positives))
    x_train, y_train = np.asarray(x_train), np.asarray(y_train)

    model = (LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, verbosity=-1, random_state=42)
             if LGBMClassifier else HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=42))
    model.fit(x_train, y_train)
    train_scores = model.predict_proba(x_train)[:, 1]
    threshold, train_score = best_threshold(y_train, train_scores)
    print(f"Training F_0.5={train_score:.4f}; selected threshold={threshold:.3f}")

    candidate_rows, matching_rows = [], []
    for _, left in test_s1.iterrows():
        rows = [(right, model.predict_proba([pair_features(left, right, vectorizer)])[0, 1]) for _, right in test_targets.iterrows()]
        rows.sort(key=lambda item: item[1], reverse=True)
        candidates = rows[:15]
        matches = [right.entity_id for right, score in candidates if score >= threshold]
        candidate_rows.append({"source1_entity_id": left.entity_id, "candidate_entity_ids": ",".join(right.entity_id for right, _ in candidates)})
        matching_rows.append({"source1_entity_id": left.entity_id, "matched_entity_ids": ",".join(matches)})

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(candidate_rows).to_csv(os.path.join(output_dir, "candidate_pairs.tsv"), sep="\t", index=False)
    pd.DataFrame(matching_rows).to_csv(os.path.join(output_dir, "matching_results.tsv"), sep="\t", index=False)
    print(f"Wrote {len(test_s1)} rows in {time.time() - started:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()
    main(args.train_dir, args.test_dir, args.output_dir)
