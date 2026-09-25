"""FastAPI dashboard for the business entity-resolution pipeline."""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import traceback
from threading import Lock
from difflib import SequenceMatcher
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"
RUNTIME_DIR = Path(os.getenv("ENTITYFORGE_RUNTIME_DIR", "/tmp/entityforge-ber" if os.getenv("VERCEL") else str(BASE_DIR)))
TEST_DIR = RUNTIME_DIR / "dataset" / "test"
OUTPUT_DIR = RUNTIME_DIR / "output"
DASHBOARD_TEMPLATE = BASE_DIR / "templates" / "dashboard.html"
REQUIRED_COLUMNS = {"entity_id", "business_name", "business_address", "country"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
PIPELINE_LOCK = Lock()

app = FastAPI(
    title="EntityForge BER Dashboard",
    description="Upload fragmented company data, run matching, and inspect the results.",
    version="1.0.0",
)


def read_tsv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t", dtype=str).fillna("")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read {path.name}: {exc}") from exc


def require_columns(frame: pd.DataFrame, columns: set[str], filename: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise HTTPException(status_code=500, detail=f"{filename} is missing columns: {sorted(missing)}")


def validate_upload(data: bytes, filename: str) -> pd.DataFrame:
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"{filename}: maximum upload size is 25 MB")
    if not filename.lower().endswith(".tsv"):
        raise HTTPException(status_code=400, detail=f"{filename}: upload a TSV file")
    try:
        from io import BytesIO
        frame = pd.read_csv(BytesIO(data), sep="\t", dtype=str).fillna("")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{filename}: invalid TSV ({exc})") from exc
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"{filename}: missing columns {sorted(missing)}")
    if frame["entity_id"].duplicated().any():
        raise HTTPException(status_code=400, detail=f"{filename}: entity_id values must be unique")
    return frame


def parse_delimited(data: bytes, filename: str) -> pd.DataFrame:
    """Read either CSV or TSV content for schema analysis."""
    from io import BytesIO

    suffix = Path(filename).suffix.lower()
    try:
        separator = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(BytesIO(data), sep=separator, dtype=str).fillna("")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{filename}: could not parse delimited data ({exc})") from exc


def detect_schema(frame: pd.DataFrame) -> dict:
    """Infer common entity columns using normalized header aliases."""
    aliases = {
        "entity_id": {"id", "entityid", "entityidentifier", "businessid", "companyid", "recordid", "uid", "code"},
        "business_name": {"name", "businessname", "companyname", "business", "company", "organization", "organisation", "entityname"},
        "business_address": {"address", "businessaddress", "companyaddress", "location", "street", "fulladdress"},
        "country": {"country", "countrycode", "nation", "market", "region"},
    }
    mapping, confidence = {}, {}
    used = set()
    for target, names in aliases.items():
        ranked = []
        for column in frame.columns:
            normalized = re.sub(r"[^a-z0-9]", "", str(column).lower())
            score = 0
            if normalized in names:
                score = 1.0
            elif any(alias in normalized or normalized in alias for alias in names):
                score = 0.7
            if score and column not in used:
                ranked.append((score, column))
        if ranked:
            score, column = max(ranked, key=lambda item: item[0])
            mapping[target] = column
            confidence[target] = round(score * 100)
            used.add(column)
        else:
            mapping[target] = None
            confidence[target] = 0
    return {"columns": [str(column) for column in frame.columns], "mapping": mapping, "confidence": confidence, "rows": len(frame)}


def analysis_plan(analyses: list[dict]) -> dict:
    total_rows = sum(item["schema"]["rows"] for item in analyses)
    model = "TF-IDF blocking + LightGBM classifier" if total_rows >= 500 else "TF-IDF blocking + pairwise similarity"
    return {
        "file_count": len(analyses),
        "total_rows": total_rows,
        "reference": analyses[0]["filename"] if analyses else None,
        "targets": [item["filename"] for item in analyses[1:]],
        "model": model,
        "stages": ["Normalize text", "Character TF-IDF blocking", "Pairwise features", "Precision-tuned F0.5 threshold"],
    }


def normalize_for_pipeline(frame: pd.DataFrame, schema: dict, source_label: str) -> pd.DataFrame:
    mapping = schema["mapping"]
    missing = [field for field in ("entity_id", "business_name", "business_address", "country") if not mapping.get(field)]
    if missing:
        raise HTTPException(status_code=400, detail=f"{source_label}: could not detect columns {missing}")
    normalized = pd.DataFrame({field: frame[mapping[field]].astype(str) for field in ("entity_id", "business_name", "business_address", "country")})
    normalized["entity_id"] = normalized["entity_id"].replace({"": pd.NA}).fillna(f"{source_label}-")
    blanks = normalized["entity_id"].eq(f"{source_label}-")
    normalized.loc[blanks, "entity_id"] = [f"{source_label}-{index}" for index in normalized.index[blanks]]
    return normalized


def run_pipeline() -> str:
    # Run in-process: on Vercel a child interpreter can't see the function's installed packages.
    import run_solution

    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            run_solution.main(str(TRAIN_DIR), str(TEST_DIR), str(OUTPUT_DIR))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Matching failed: {exc}") from exc
    return output.getvalue().strip() or "Pipeline completed"


def csv_ids(value: object) -> list[str]:
    if not value or pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def dashboard_rows() -> tuple[list[dict], dict]:
    source1 = read_tsv(TEST_DIR / "test_source1.tsv")
    source2_frame = read_tsv(TEST_DIR / "test_source2.tsv")
    source3_frame = read_tsv(TEST_DIR / "test_source3.tsv")
    source2 = pd.concat([source2_frame, source3_frame], ignore_index=True)
    matches = read_tsv(OUTPUT_DIR / "matching_results.tsv")
    candidates = read_tsv(OUTPUT_DIR / "candidate_pairs.tsv")
    require_columns(source1, REQUIRED_COLUMNS, "test_source1.tsv")
    require_columns(source2, REQUIRED_COLUMNS, "test_source2.tsv/test_source3.tsv")
    require_columns(matches, {"source1_entity_id", "matched_entity_ids"}, "matching_results.tsv")
    require_columns(candidates, {"source1_entity_id", "candidate_entity_ids"}, "candidate_pairs.tsv")
    if source2["entity_id"].duplicated().any():
        raise HTTPException(status_code=500, detail="Source 2 and Source 3 contain duplicate entity IDs")
    lookup = source2.set_index("entity_id").to_dict("index")
    candidate_lookup = candidates.set_index("source1_entity_id")["candidate_entity_ids"].to_dict()
    match_lookup = matches.set_index("source1_entity_id")["matched_entity_ids"].to_dict()

    rows = []
    for record in source1.to_dict("records"):
        source_id = record["entity_id"]
        matched = csv_ids(match_lookup.get(source_id, ""))
        details = []
        for item in matched:
            if item not in lookup:
                continue
            detail = dict(lookup[item])
            name_score = SequenceMatcher(None, str(record["business_name"]).lower(), str(detail["business_name"]).lower()).ratio()
            address_score = SequenceMatcher(None, str(record["business_address"]).lower(), str(detail["business_address"]).lower()).ratio()
            detail["source"] = item.split("-", 1)[0]
            detail["confidence"] = round((name_score * 0.6 + address_score * 0.4) * 100)
            details.append(detail)
        rows.append({
            "source1": record,
            "matched_ids": matched,
            "matched": details,
            "candidates": csv_ids(candidate_lookup.get(source_id, "")),
        })
    confidences = [match["confidence"] for row in rows for match in row["matched"]]
    summary = {
        "total": len(rows),
        "matched": sum(bool(row["matched_ids"]) for row in rows),
        "singletons": sum(not row["matched_ids"] for row in rows),
        "mean_confidence": round(sum(confidences) / len(confidences)) if confidences else 0,
        "source2_matches": sum(1 for row in rows for match in row["matched"] if match["source"] == "S2"),
        "source3_matches": sum(1 for row in rows for match in row["matched"] if match["source"] == "S3"),
    }
    return rows, summary


def render_dashboard(message: str = "") -> str:
    try:
        rows, summary = dashboard_rows()
    except Exception:
        rows, summary = [], {"total": 0, "matched": 0, "singletons": 0, "mean_confidence": 0, "source2_matches": 0, "source3_matches": 0}
    payload = (json.dumps({"rows": rows, "summary": summary, "notice": message}, ensure_ascii=False)
               .replace("<", "\\u003c")
               .replace(">", "\\u003e")
               .replace("&", "\\u0026"))
    return DASHBOARD_TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", payload)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return render_dashboard()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze/", response_class=JSONResponse)
async def analyze_files(files: Annotated[list[UploadFile], File(...)]) -> JSONResponse:
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Upload at least two CSV or TSV files for analysis")
    analyses = []
    for upload in files:
        filename = upload.filename or "uploaded.tsv"
        data = await upload.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"{filename}: maximum upload size is 25 MB")
        if Path(filename).suffix.lower() not in {".csv", ".tsv"}:
            raise HTTPException(status_code=400, detail=f"{filename}: upload a CSV or TSV file")
        frame = parse_delimited(data, filename)
        schema = detect_schema(frame)
        role = "reference" if not analyses else "target"
        analyses.append({"filename": filename, "role": role, "schema": schema})
    return JSONResponse({"files": analyses, "plan": analysis_plan(analyses)})


@app.post("/smart-run/", response_class=JSONResponse)
async def smart_run(files: Annotated[list[UploadFile], File(...)]) -> JSONResponse:
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Upload at least two CSV or TSV files for smart matching")
    from io import BytesIO

    normalized = []
    target_frames = []
    seen_ids: set[str] = set()
    for index, upload in enumerate(files):
        filename = upload.filename or f"upload-{index}.tsv"
        data = await upload.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"{filename}: maximum upload size is 25 MB")
        frame = parse_delimited(data, filename)
        schema = detect_schema(frame)
        label = "reference" if index == 0 else f"target{index}"
        canonical = normalize_for_pipeline(frame, schema, label)
        if canonical["entity_id"].duplicated().any():
            raise HTTPException(status_code=400, detail=f"{filename}: ID values must be unique within the file")
        # Files exported separately often reuse the same IDs (1, 2, 3…); namespace a
        # comparison file by its name when its IDs clash with an earlier file.
        if index and seen_ids.intersection(canonical["entity_id"]):
            canonical["entity_id"] = Path(filename).stem + ":" + canonical["entity_id"]
        seen_ids.update(canonical["entity_id"])
        if index == 0:
            normalized.append(canonical)
        else:
            target_frames.append(canonical)
    targets = pd.concat(target_frames, ignore_index=True)
    normalized.append(targets)
    normalized.append(pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"]))
    uploads = [UploadFile(filename=f"smart-source{index + 1}.tsv", file=BytesIO(frame.to_csv(sep="\t", index=False).encode("utf-8"))) for index, frame in enumerate(normalized)]
    log = await execute_uploads(uploads[0], uploads[1], uploads[2])
    rows, summary = dashboard_rows()
    return JSONResponse({"message": log, "rows": rows, "summary": summary, "downloads": {"matching": "/download/matching", "candidates": "/download/candidates"}})


def snapshot_files(paths: list[Path]) -> dict[Path, bytes | None]:
    return {path: path.read_bytes() if path.exists() else None for path in paths}


def restore_files(snapshot: dict[Path, bytes | None]) -> None:
    for path, content in snapshot.items():
        if content is None:
            if path.exists():
                path.unlink()
        else:
            path.write_bytes(content)


async def execute_uploads(
    source1: Annotated[UploadFile, File(...)],
    source2: Annotated[UploadFile, File(...)],
    source3: Annotated[UploadFile, File(...)],
) -> str:
    uploads = [(source1, "test_source1.tsv"), (source2, "test_source2.tsv"), (source3, "test_source3.tsv")]
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, bytes]] = []
    seen_ids: set[str] = set()
    for upload, destination in uploads:
        data = await upload.read()
        frame = validate_upload(data, upload.filename or destination)
        duplicate_ids = seen_ids.intersection(frame["entity_id"].astype(str))
        if duplicate_ids:
            raise HTTPException(status_code=400, detail=f"Duplicate entity IDs across uploads: {sorted(duplicate_ids)}")
        seen_ids.update(frame["entity_id"].astype(str))
        staged.append((TEST_DIR / destination, data))
    watched = [path for path, _ in staged] + [OUTPUT_DIR / "matching_results.tsv", OUTPUT_DIR / "candidate_pairs.tsv"]
    with PIPELINE_LOCK:
        snapshot = snapshot_files(watched)
        try:
            # Validate all three files before replacing any existing dataset file.
            for destination, data in staged:
                if destination.exists() and destination.read_bytes() == data:
                    continue
                temporary = destination.with_suffix(destination.suffix + ".uploading")
                temporary.write_bytes(data)
                temporary.replace(destination)
            pipeline_output = run_pipeline()
        except Exception:
            restore_files(snapshot)
            raise
    return pipeline_output.splitlines()[-1] if pipeline_output else "Pipeline completed"


@app.post("/process", response_class=HTMLResponse)
async def process(
    source1: Annotated[UploadFile, File(...)],
    source2: Annotated[UploadFile, File(...)],
    source3: Annotated[UploadFile, File(...)],
) -> str:
    log = await execute_uploads(source1, source2, source3)
    return render_dashboard(f"Matching job completed: {log}")


@app.post("/upload-and-run/", response_class=JSONResponse, include_in_schema=False)
async def upload_and_run(
    source1: Annotated[UploadFile, File(...)],
    source2: Annotated[UploadFile, File(...)],
    source3: Annotated[UploadFile, File(...)],
) -> JSONResponse:
    log = await execute_uploads(source1, source2, source3)
    rows, summary = dashboard_rows()
    return JSONResponse({
        "message": log,
        "rows": rows,
        "summary": summary,
        "downloads": {"matching": "/download/matching", "candidates": "/download/candidates"},
    })


@app.get("/download/matching")
@app.get("/download/matching-results", include_in_schema=False)
def download_matching() -> FileResponse:
    path = OUTPUT_DIR / "matching_results.tsv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Matching results are not available yet")
    return FileResponse(path, filename=path.name, media_type="text/tab-separated-values")


@app.get("/download/candidates")
@app.get("/download/candidate-pairs", include_in_schema=False)
def download_candidates() -> FileResponse:
    path = OUTPUT_DIR / "candidate_pairs.tsv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Candidate pairs are not available yet")
    return FileResponse(path, filename=path.name, media_type="text/tab-separated-values")
