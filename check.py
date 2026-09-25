"""Convenience checker for the generated submission files."""
from utils.validate_submission import validate


if __name__ == "__main__":
    validate("output/matching_results.tsv", "output/candidate_pairs.tsv", "dataset/test")