import sys
import argparse
import pandas as pd

def validate(matching_file, candidate_file, test_dir):
    print("Running Submission Validation Checklist...")
    try:
        cand_df = pd.read_csv(candidate_file, sep="\t")
        match_df = pd.read_csv(matching_file, sep="\t")
        s1_df = pd.read_csv(f"{test_dir}/test_source1.tsv", sep="\t")
        
        # Check columns
        assert "source1_entity_id" in cand_df.columns, "candidate_pairs.tsv missing 'source1_entity_id'"
        assert "candidate_entity_ids" in cand_df.columns, "candidate_pairs.tsv missing 'candidate_entity_ids'"
        assert "source1_entity_id" in match_df.columns, "matching_results.tsv missing 'source1_entity_id'"
        assert "matched_entity_ids" in match_df.columns, "matching_results.tsv missing 'matched_entity_ids'"
        
        # Check row count
        assert len(cand_df) == len(s1_df), f"Candidate row count ({len(cand_df)}) does not match Source 1 count ({len(s1_df)})"
        assert len(match_df) == len(s1_df), f"Matching row count ({len(match_df)}) does not match Source 1 count ({len(s1_df)})"

        valid_source_ids = set(s1_df["entity_id"].astype(str))
        assert set(cand_df["source1_entity_id"].astype(str)) == valid_source_ids, "Candidate Source 1 IDs do not match test Source 1"
        assert set(match_df["source1_entity_id"].astype(str)) == valid_source_ids, "Matching Source 1 IDs do not match test Source 1"
        for column in ("candidate_entity_ids", "matched_entity_ids"):
            frame = cand_df if column == "candidate_entity_ids" else match_df
            for value in frame[column].fillna(""):
                ids = [item.strip() for item in str(value).split(",") if item.strip()]
                assert len(ids) == len(set(ids)), f"Duplicate IDs in {column}"
                assert all(item not in valid_source_ids for item in ids), f"Self-match in {column}"
        
        print("✅ PASS: All files and formats are 100% valid!")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--test-dir", required=True)
    args = parser.parse_args()
    validate(args.matching, args.candidate, args.test_dir)
