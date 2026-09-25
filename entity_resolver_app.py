"""EntityForge — FastAPI app serving the entity-resolution dashboard and API.

One reference file is matched against one or more comparison files. The ML
pipeline (``run_solution.resolve``) runs in-process, which keeps it working on
Vercel, where a child Python process can't see the function's packages.
"""
from __future__ import annotations

import io
import json
import os
import re
import traceback
from pathlib import Path
from threading import Lock
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse

# Heavy scientific imports are guarded so the page and /health still load and can
# report a readable error if the deployment is missing a dependency.
try:
    import pandas as pd
    import run_solution as pipeline
    IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - depends on the deployment
    pd = pipeline = None
    IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"
SAMPLE_DIR = BASE_DIR / "dataset" / "test"
# Vercel's filesystem is read-only except /tmp.
DATA_ROOT = Path(os.getenv("ENTITYFORGE_RUNTIME_DIR") or ("/tmp" if os.getenv("VERCEL") else BASE_DIR / ".runtime"))
UPLOAD_DIR = DATA_ROOT / "uploads"
OUTPUT_DIR = DATA_ROOT / "output"
REQUIRED_COLUMNS = ("entity_id", "business_name", "business_address", "country")
ALLOWED_SUFFIXES = {".tsv", ".csv"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
OUTPUT_FILES = {"matching": "matching_results.tsv", "candidates": "candidate_pairs.tsv"}
RUN_LOCK = Lock()

app = FastAPI(
    title="EntityForge",
    description="Match business records from comparison sources against a reference list.",
    version="2.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def safe_name(filename: str | None, fallback: str) -> str:
    name = Path(filename or fallback).name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name) or fallback


def parse_table(data: bytes, filename: str) -> "pd.DataFrame":
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"{filename}: upload a .tsv or .csv file.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{filename}: files can be at most 25 MB.")
    try:
        frame = pd.read_csv(io.BytesIO(data), sep="\t" if suffix == ".tsv" else ",", dtype=str,
                            encoding="utf-8-sig", keep_default_na=False)
    except Exception as exc:
        raise HTTPException(400, f"{filename}: could not read the file ({exc}).") from exc
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise HTTPException(400, f"{filename} is missing {', '.join(missing)}. "
                                 f"Expected headers: {', '.join(REQUIRED_COLUMNS)}.")
    frame = frame[list(REQUIRED_COLUMNS)].apply(lambda column: column.str.strip())
    frame = frame[frame.ne("").any(axis=1)].reset_index(drop=True)
    if frame.empty:
        raise HTTPException(400, f"{filename} has no records.")
    if frame["entity_id"].eq("").any():
        raise HTTPException(400, f"{filename}: every row needs an entity_id.")
    duplicated = frame["entity_id"][frame["entity_id"].duplicated()]
    if not duplicated.empty:
        raise HTTPException(400, f"{filename}: entity_id must be unique (repeated: {duplicated.iloc[0]}).")
    return frame


async def read_upload(upload: UploadFile, fallback: str) -> tuple[str, bytes]:
    name = safe_name(upload.filename, fallback)
    data = await upload.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{name}: files can be at most 25 MB.")
    return name, data


def pool_targets(reference: "pd.DataFrame", targets: list[tuple[str, "pd.DataFrame"]]):
    """Stack comparison files, namespacing IDs that clash with earlier files."""
    seen = set(reference["entity_id"])
    frames, sources = [], {}
    for name, frame in targets:
        frame = frame.copy()
        if seen.intersection(frame["entity_id"]):
            frame["entity_id"] = Path(name).stem + ":" + frame["entity_id"]
        seen.update(frame["entity_id"])
        sources.update(dict.fromkeys(frame["entity_id"], name))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), sources


# ---------------------------------------------------------------------------
# Matching job
# ---------------------------------------------------------------------------

def run_job(reference_file: tuple[str, bytes], target_files: list[tuple[str, bytes]]) -> dict:
    if IMPORT_ERROR:
        raise HTTPException(503, f"The matching engine could not start ({IMPORT_ERROR}).")
    if not target_files:
        raise HTTPException(400, "Add at least one comparison file.")
    ref_name, ref_data = reference_file
    reference = parse_table(ref_data, ref_name)
    targets = [(name, parse_table(data, name)) for name, data in target_files]
    pooled, sources = pool_targets(reference, targets)

    with RUN_LOCK:
        try:
            result = pipeline.resolve(reference, pooled, str(TRAIN_DIR))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(500, f"Matching failed: {exc}") from exc
        matching_tsv = result["matches"].to_csv(sep="\t", index=False)
        candidates_tsv = result["candidates"].to_csv(sep="\t", index=False)
        save_run(reference_file, target_files, matching_tsv, candidates_tsv)

    return build_payload(ref_name, reference, targets, pooled, sources, result, matching_tsv, candidates_tsv)


def save_run(reference_file, target_files, matching_tsv: str, candidates_tsv: str) -> None:
    """Keep the latest inputs and outputs in /tmp (best effort; the response carries everything)."""
    try:
        for directory in (UPLOAD_DIR, OUTPUT_DIR):
            directory.mkdir(parents=True, exist_ok=True)
        for old in UPLOAD_DIR.iterdir():
            if old.is_file():
                old.unlink()
        (UPLOAD_DIR / f"reference__{reference_file[0]}").write_bytes(reference_file[1])
        for index, (name, data) in enumerate(target_files, start=1):
            (UPLOAD_DIR / f"target{index}__{name}").write_bytes(data)
        (OUTPUT_DIR / OUTPUT_FILES["matching"]).write_text(matching_tsv, encoding="utf-8")
        (OUTPUT_DIR / OUTPUT_FILES["candidates"]).write_text(candidates_tsv, encoding="utf-8")
    except OSError:
        traceback.print_exc()


def build_payload(ref_name, reference, targets, pooled, sources, result, matching_tsv, candidates_tsv) -> dict:
    lookup = {row["entity_id"]: row for row in pooled.to_dict("records")}
    threshold = result["threshold"]

    def entity(entity_id: str, probability: float) -> dict:
        row = lookup[entity_id]
        return {"id": entity_id, "name": row["business_name"], "address": row["business_address"],
                "country": row["country"], "source": sources[entity_id], "score": round(probability * 100)}

    rows, links_per_source, scores = [], dict.fromkeys((name for name, _ in targets), 0), []
    for record in reference.to_dict("records"):
        pairs = result["scores"].get(record["entity_id"], [])
        matches = [entity(e, p) for e, p in pairs if p >= threshold]
        for match in matches:
            links_per_source[match["source"]] += 1
            scores.append(match["score"])
        rows.append({
            "id": record["entity_id"], "name": record["business_name"], "address": record["business_address"],
            "country": record["country"], "matches": matches,
            "nearest": None if matches or not pairs else entity(*pairs[0]),
            "candidates": len(pairs),
        })

    matched = sum(bool(row["matches"]) for row in rows)
    return {
        "rows": rows,
        "summary": {
            "reference": {"name": ref_name, "records": len(reference)},
            "sources": [{"name": name, "records": len(frame), "links": links_per_source[name]} for name, frame in targets],
            "target_records": len(pooled),
            "matched": matched,
            "unmatched": len(rows) - matched,
            "links": len(scores),
            "avg_score": round(sum(scores) / len(scores)) if scores else 0,
            "model": result["model"],
            "threshold": round(threshold, 3),
            "seconds": round(result["seconds"], 2),
        },
        "downloads": {"matching": matching_tsv, "candidates": candidates_tsv},
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home() -> str:
    config = {
        "model": getattr(pipeline, "MODEL_NAME", "gradient boosting"),
        "engineError": IMPORT_ERROR,
        "sample": (SAMPLE_DIR / "test_source1.tsv").exists(),
    }
    payload = json.dumps(config).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return DASHBOARD_HTML.replace("__CONFIG__", payload)


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if not IMPORT_ERROR else "degraded",
            "model": getattr(pipeline, "MODEL_NAME", None), "error": IMPORT_ERROR}


@app.post("/api/resolve")
async def resolve_files(
    reference: Annotated[UploadFile, File(description="Reference source (Source 1)")],
    targets: Annotated[list[UploadFile] | None, File(description="One or more comparison sources")] = None,
) -> dict:
    reference_file = await read_upload(reference, "reference.tsv")
    target_files, names = [], {reference_file[0]}
    for index, upload in enumerate(targets or [], start=1):
        name, data = await read_upload(upload, f"comparison{index}.tsv")
        if name in names:  # keep per-file labels distinct
            name = f"{Path(name).stem}-{index}{Path(name).suffix}"
        names.add(name)
        target_files.append((name, data))
    return await run_in_threadpool(run_job, reference_file, target_files)


@app.post("/api/resolve/sample")
async def resolve_sample() -> dict:
    names = ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]
    if not all((SAMPLE_DIR / name).exists() for name in names):
        raise HTTPException(404, "Sample data is not bundled with this deployment.")
    files = [(name, (SAMPLE_DIR / name).read_bytes()) for name in names]
    return await run_in_threadpool(run_job, files[0], files[1:])


@app.get("/download/{kind}")
def download(kind: str) -> FileResponse:
    if kind not in OUTPUT_FILES:
        raise HTTPException(404, "Unknown file.")
    path = OUTPUT_DIR / OUTPUT_FILES[kind]
    if not path.exists():
        raise HTTPException(404, "Run a match first — no results are stored on this server instance.")
    return FileResponse(path, filename=path.name, media_type="text/tab-separated-values")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>EntityForge — Entity resolution</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0f172a;
  --card: rgba(30, 41, 59, .62);
  --card-solid: #1e293b;
  --inset: rgba(15, 23, 42, .55);
  --line: rgba(148, 163, 184, .14);
  --line-strong: rgba(148, 163, 184, .26);
  --text: #e2e8f0;
  --strong: #f8fafc;
  --muted: #94a3b8;
  --faint: #64748b;
  --primary: #6366f1;
  --primary-hover: #5558e6;
  --primary-soft: rgba(99, 102, 241, .14);
  --primary-text: #c7d2fe;
  --emerald: #059669;
  --emerald-soft: rgba(5, 150, 105, .15);
  --emerald-line: rgba(5, 150, 105, .38);
  --emerald-text: #8fd9bb;
  --danger-soft: rgba(239, 68, 68, .1);
  --danger-line: rgba(239, 68, 68, .28);
  --danger-text: #f3b0b0;
  --radius: 14px;
  --ease: cubic-bezier(.22, .8, .24, 1);
  --sans: "Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
}
* { box-sizing: border-box; }
html { background: var(--bg); -webkit-text-size-adjust: 100%; }
html, body { margin: 0; padding: 0; min-height: 100%; }
body {
  min-height: 100vh; overflow-x: hidden; color: var(--text); font: 15px/1.55 var(--sans);
  background: radial-gradient(1200px 520px at 50% -180px, #1e293b 0%, rgba(15, 23, 42, 0) 70%), var(--bg);
  -webkit-font-smoothing: antialiased;
}
button, input { font: inherit; color: inherit; }
[hidden] { display: none !important; }
.mono { font-family: var(--mono); font-size: .82em; letter-spacing: -.01em; }
.muted { color: var(--muted); }
.sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }

.shell { width: 100%; max-width: 1120px; margin: 0 auto; padding: 0 24px 96px; }

/* Header */
.topbar { display: flex; align-items: center; justify-content: space-between; height: 72px; }
.brand { display: flex; align-items: center; gap: 10px; font-weight: 600; color: var(--strong); text-decoration: none; letter-spacing: -.01em; }
.brand-mark { display: grid; place-items: center; width: 30px; height: 30px; border-radius: 8px; background: var(--card-solid); border: 1px solid var(--line); }
.brand-mark svg { width: 16px; height: 16px; color: var(--primary-text); }
.engine { font-size: 13px; color: var(--faint); }
.engine b { font-weight: 500; color: var(--muted); }

.intro { padding: 40px 0 32px; max-width: 640px; }
.intro h1 { margin: 0 0 12px; font-size: clamp(28px, 4vw, 40px); line-height: 1.12; letter-spacing: -.03em; font-weight: 700; color: var(--strong); }
.intro p { margin: 0; color: var(--muted); font-size: 16px; }

/* Cards */
.card {
  background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  box-shadow: 0 1px 0 rgba(255, 255, 255, .03) inset, 0 20px 40px -24px rgba(2, 6, 23, .7);
}

/* Stepper */
.steps { display: flex; list-style: none; margin: 0; padding: 0 28px; border-bottom: 1px solid var(--line); gap: 32px; overflow-x: auto; }
.steps li { position: relative; display: flex; align-items: center; gap: 10px; padding: 18px 0; font-size: 14px; color: var(--faint); white-space: nowrap; transition: color .3s var(--ease); }
.steps li::after { content: ""; position: absolute; left: 0; right: 0; bottom: -1px; height: 2px; border-radius: 2px; background: var(--primary); transform: scaleX(0); transform-origin: left; transition: transform .5s var(--ease); }
.steps li.current { color: var(--strong); }
.steps li.current::after { transform: scaleX(1); }
.steps li.done { color: var(--muted); }
.step-dot { display: grid; place-items: center; width: 24px; height: 24px; border-radius: 50%; border: 1px solid var(--line-strong); font-size: 12px; font-weight: 600; transition: all .35s var(--ease); }
.steps li.current .step-dot { border-color: var(--primary); color: var(--primary-text); background: var(--primary-soft); }
.steps li.done .step-dot { border-color: var(--emerald); background: var(--emerald); color: #fff; }
.step-dot svg { width: 12px; height: 12px; }

/* Step panels */
.panel { padding: 28px; }
.panel.entering { animation: stepIn .45s var(--ease) both; }
.panel.leaving { animation: stepOut .2s ease both; }
.panel-head { margin-bottom: 18px; }
.panel-head h2 { margin: 0 0 4px; font-size: 17px; font-weight: 600; color: var(--strong); letter-spacing: -.01em; }
.panel-head p { margin: 0; font-size: 14px; color: var(--muted); }
.panel-head code { font-family: var(--mono); font-size: 12px; color: var(--text); background: var(--inset); border: 1px solid var(--line); padding: 1px 6px; border-radius: 5px; }

.drop {
  position: relative; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 4px;
  min-height: 168px; padding: 28px 20px; border: 1.5px dashed var(--line-strong); border-radius: 12px; background: var(--inset);
  text-align: center; cursor: pointer; transition: border-color .25s, background .25s, transform .3s var(--ease);
}
.drop:hover { border-color: rgba(148, 163, 184, .45); }
.drop.dragging { border-color: var(--primary); background: var(--primary-soft); transform: scale(1.006); }
.drop input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
.drop-icon { display: grid; place-items: center; width: 44px; height: 44px; margin-bottom: 8px; border-radius: 12px; background: var(--card-solid); border: 1px solid var(--line); transition: transform .35s var(--ease); }
.drop.dragging .drop-icon, .drop:hover .drop-icon { transform: translateY(-3px); }
.drop-icon svg { width: 20px; height: 20px; color: var(--muted); }
.drop strong { font-weight: 500; color: var(--strong); }
.drop small { color: var(--faint); font-size: 13px; }

.file-list { list-style: none; margin: 14px 0 0; padding: 0; display: grid; gap: 8px; }
.file {
  display: grid; grid-template-columns: 36px 1fr auto; gap: 12px; align-items: center; padding: 12px 14px;
  border: 1px solid var(--line); border-radius: 10px; background: var(--card-solid); animation: rise .35s var(--ease) both;
}
.file.bad { border-color: var(--danger-line); background: var(--danger-soft); }
.file.leaving { animation: fadeAway .2s ease forwards; }
.file-icon { display: grid; place-items: center; width: 36px; height: 36px; border-radius: 8px; background: var(--inset); color: var(--muted); }
.file-icon svg { width: 18px; height: 18px; }
.file.ok .file-icon { color: var(--emerald-text); background: var(--emerald-soft); }
.file-name { font-weight: 500; color: var(--strong); word-break: break-all; }
.file-meta { font-size: 13px; color: var(--muted); }
.file.bad .file-meta { color: var(--danger-text); }
.icon-btn { display: grid; place-items: center; width: 32px; height: 32px; border: 0; border-radius: 8px; background: transparent; color: var(--faint); cursor: pointer; transition: background .2s, color .2s; }
.icon-btn:hover { background: var(--inset); color: var(--text); }
.icon-btn svg { width: 16px; height: 16px; }

.ref-chip { display: flex; align-items: center; gap: 12px; padding: 12px 14px; margin-bottom: 16px; border-radius: 10px; background: var(--primary-soft); border: 1px solid rgba(99, 102, 241, .28); font-size: 14px; }
.ref-chip .tag { font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--primary-text); }
.ref-chip .grow { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--strong); }
.link-btn { border: 0; background: none; padding: 0; color: var(--primary-text); font-size: 14px; cursor: pointer; text-decoration: underline; text-underline-offset: 3px; text-decoration-color: rgba(199, 210, 254, .35); transition: text-decoration-color .2s; }
.link-btn:hover { text-decoration-color: currentColor; }

.actions { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-top: 22px; flex-wrap: wrap; }
.actions .right { display: flex; gap: 10px; margin-left: auto; }
.btn {
  display: inline-flex; align-items: center; justify-content: center; gap: 8px; height: 40px; padding: 0 18px;
  border-radius: 10px; border: 1px solid transparent; font-size: 14px; font-weight: 500; white-space: nowrap;
  text-decoration: none; cursor: pointer; transition: background .2s, border-color .2s, transform .15s var(--ease), box-shadow .2s, opacity .2s;
}
.btn svg { width: 16px; height: 16px; }
.btn:active { transform: translateY(1px); }
.btn:focus-visible, .icon-btn:focus-visible, .link-btn:focus-visible, .seg button:focus-visible { outline: 2px solid var(--primary); outline-offset: 2px; }
.btn-primary { background: var(--primary); color: #fff; box-shadow: 0 1px 0 rgba(255, 255, 255, .12) inset; }
.btn-primary:hover { background: var(--primary-hover); }
.btn-ghost { background: transparent; border-color: var(--line-strong); color: var(--text); }
.btn-ghost:hover { background: var(--inset); border-color: rgba(148, 163, 184, .4); }
.btn[disabled] { opacity: .4; cursor: not-allowed; transform: none; }

.alert { margin-top: 14px; padding: 12px 14px; border-radius: 10px; background: var(--danger-soft); border: 1px solid var(--danger-line); color: var(--danger-text); font-size: 14px; animation: rise .3s var(--ease) both; }

/* Running */
.run-state { display: grid; gap: 22px; }
.progress { height: 6px; border-radius: 6px; background: var(--inset); overflow: hidden; border: 1px solid var(--line); }
.progress i { display: block; height: 100%; width: 0; border-radius: 6px; background: linear-gradient(90deg, #4f46e5, var(--primary)); transition: width .5s var(--ease); }
.stages { list-style: none; margin: 0; padding: 0; display: grid; gap: 12px; }
.stage { display: flex; align-items: center; gap: 12px; font-size: 14px; color: var(--faint); transition: color .3s; }
.stage-icon { display: grid; place-items: center; width: 22px; height: 22px; flex: none; border-radius: 50%; border: 1px solid var(--line-strong); }
.stage-icon svg { width: 12px; height: 12px; opacity: 0; transform: scale(.4); transition: all .3s var(--ease); }
.stage.active { color: var(--strong); }
.stage.active .stage-icon { border-color: transparent; border-top-color: var(--primary); border-right-color: var(--primary); animation: spin .8s linear infinite; }
.stage.done { color: var(--muted); }
.stage.done .stage-icon { background: var(--emerald); border-color: var(--emerald); color: #fff; }
.stage.done .stage-icon svg { opacity: 1; transform: none; }
.elapsed { font-size: 13px; color: var(--faint); font-variant-numeric: tabular-nums; }

.done-state { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
.done-badge { display: grid; place-items: center; width: 44px; height: 44px; border-radius: 12px; background: var(--emerald-soft); color: var(--emerald-text); animation: pop .5s var(--ease) both; }
.done-badge svg { width: 22px; height: 22px; }
.done-state .grow { flex: 1; min-width: 200px; }

/* Results */
.results { margin-top: 56px; }
.results.entering { animation: rise .6s var(--ease) both; }
.results-head { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; margin-bottom: 18px; flex-wrap: wrap; }
.results-head h2 { margin: 0; font-size: 22px; letter-spacing: -.02em; color: var(--strong); }
.results-head p { margin: 4px 0 0; color: var(--muted); font-size: 14px; }
.downloads { display: flex; gap: 10px; flex-wrap: wrap; }

.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 14px; }
.stat { padding: 20px; transition: transform .3s var(--ease), border-color .3s, box-shadow .3s; animation: rise .5s var(--ease) both; }
.stat:hover { transform: translateY(-3px); border-color: var(--line-strong); box-shadow: 0 24px 40px -24px rgba(2, 6, 23, .9); }
.stat-label { font-size: 13px; color: var(--muted); }
.stat-value { display: block; margin: 6px 0 2px; font-size: 30px; font-weight: 600; letter-spacing: -.03em; color: var(--strong); font-variant-numeric: tabular-nums; }
.stat-foot { font-size: 12px; color: var(--faint); }
@property --n { syntax: "<integer>"; initial-value: 0; inherits: false; }
.count { --n: var(--to); counter-reset: n var(--n); animation: count 1s var(--ease) .15s both; }
.count::after { content: counter(n) attr(data-suffix); }
@keyframes count { from { --n: 0; } }

.sources { padding: 18px 20px; margin-bottom: 28px; display: grid; gap: 12px; }
.sources h3 { margin: 0 0 2px; font-size: 13px; font-weight: 500; color: var(--muted); }
.source-row { display: grid; grid-template-columns: minmax(120px, 220px) 1fr auto; gap: 14px; align-items: center; font-size: 13px; }
.source-row .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text); }
.track { height: 6px; border-radius: 6px; background: var(--inset); overflow: hidden; }
.track i { display: block; height: 100%; width: 0; border-radius: 6px; background: var(--emerald); transition: width 1s var(--ease); }
.source-row .num { color: var(--muted); font-variant-numeric: tabular-nums; white-space: nowrap; }

.toolbar { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 14px; flex-wrap: wrap; }
.seg { position: relative; display: inline-flex; padding: 4px; border-radius: 10px; background: var(--inset); border: 1px solid var(--line); }
.seg button { position: relative; z-index: 1; border: 0; background: none; padding: 6px 14px; border-radius: 7px; font-size: 13px; color: var(--muted); cursor: pointer; transition: color .2s; }
.seg button.active { color: var(--strong); }
.seg .thumb { position: absolute; top: 4px; bottom: 4px; left: 0; border-radius: 7px; background: var(--card-solid); border: 1px solid var(--line-strong); transition: transform .4s var(--ease), width .4s var(--ease); }
.search { position: relative; flex: 0 1 320px; }
.search svg { position: absolute; left: 12px; top: 50%; width: 16px; height: 16px; transform: translateY(-50%); color: var(--faint); pointer-events: none; }
.search input { width: 100%; height: 40px; padding: 0 12px 0 38px; border-radius: 10px; border: 1px solid var(--line); background: var(--inset); outline: none; font-size: 14px; transition: border-color .2s, box-shadow .2s; }
.search input::placeholder { color: var(--faint); }
.search input:focus { border-color: rgba(99, 102, 241, .6); box-shadow: 0 0 0 3px var(--primary-soft); }

.pairs { display: grid; gap: 12px; }
.pair { display: grid; grid-template-columns: 1fr 36px 1fr; gap: 10px; align-items: start; padding: 14px; animation: rise .4s var(--ease) both; }
.connector { display: grid; place-items: center; align-self: center; width: 36px; height: 36px; border-radius: 50%; background: var(--inset); border: 1px solid var(--line); color: var(--faint); }
.connector svg { width: 16px; height: 16px; }
.pair.matched .connector { color: var(--emerald-text); border-color: var(--emerald-line); background: var(--emerald-soft); }
.stack { display: grid; gap: 10px; }
.entity { padding: 14px 16px; border-radius: 10px; background: var(--card-solid); border: 1px solid var(--line); transition: transform .3s var(--ease), border-color .3s, box-shadow .3s; }
.entity:hover { transform: translateY(-2px); border-color: var(--line-strong); box-shadow: 0 18px 30px -20px rgba(2, 6, 23, .95); }
.entity-top { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 6px; font-size: 11px; color: var(--faint); }
.entity-top .label { text-transform: uppercase; letter-spacing: .07em; font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.entity-name { font-weight: 600; color: var(--strong); letter-spacing: -.005em; }
.entity-addr { font-size: 13px; color: var(--muted); margin-top: 2px; }
.entity-foot { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-top: 10px; flex-wrap: wrap; }
.country { font-size: 12px; color: var(--muted); padding: 2px 8px; border-radius: 6px; background: var(--inset); border: 1px solid var(--line); }
.score { font-size: 12px; font-weight: 600; padding: 3px 9px; border-radius: 99px; white-space: nowrap; }
.score.match { color: var(--emerald-text); background: var(--emerald-soft); border: 1px solid var(--emerald-line); }
.score.weak { color: var(--muted); background: var(--inset); border: 1px solid var(--line); font-weight: 500; }
.entity.empty { background: transparent; border-style: dashed; }
.entity.empty:hover { transform: none; box-shadow: none; }
.no-match { font-size: 14px; color: var(--muted); }
.nearest { margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--line); font-size: 13px; color: var(--faint); }
.nearest b { font-weight: 500; color: var(--text); }
.list-empty { padding: 48px 20px; text-align: center; color: var(--muted); }
.more { display: flex; justify-content: center; margin-top: 18px; }

.toast { position: fixed; left: 50%; bottom: 24px; z-index: 20; max-width: calc(100% - 32px); padding: 11px 16px; border-radius: 10px; background: var(--card-solid); border: 1px solid var(--line-strong); color: var(--strong); font-size: 14px; box-shadow: 0 20px 40px -16px rgba(2, 6, 23, .9); opacity: 0; transform: translate(-50%, 16px); transition: opacity .3s, transform .45s var(--ease); pointer-events: none; }
.toast.show { opacity: 1; transform: translate(-50%, 0); }

.reveal { animation: rise .7s var(--ease) both; }
.reveal.d1 { animation-delay: .08s; }
.reveal.d2 { animation-delay: .16s; }

@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
@keyframes stepIn { from { opacity: 0; transform: translateX(14px); } to { opacity: 1; transform: none; } }
@keyframes stepOut { to { opacity: 0; transform: translateX(-10px); } }
@keyframes fadeAway { to { opacity: 0; transform: translateX(10px); } }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes pop { 0% { opacity: 0; transform: scale(.6); } 70% { transform: scale(1.06); } 100% { opacity: 1; transform: none; } }

@media (max-width: 860px) {
  .stats { grid-template-columns: repeat(2, 1fr); }
  .pair { grid-template-columns: 1fr; }
  .connector { justify-self: center; transform: rotate(90deg); }
}
@media (max-width: 600px) {
  .shell { padding: 0 16px 64px; }
  .engine { display: none; }
  .intro { padding: 24px 0 24px; }
  .steps { padding: 0 18px; gap: 22px; }
  .panel { padding: 20px 18px; }
  .stats { gap: 10px; }
  .stat { padding: 16px; }
  .stat-value { font-size: 24px; }
  .source-row { grid-template-columns: 1fr auto; }
  .source-row .track { grid-column: 1 / -1; grid-row: 2; }
  .search { flex: 1 1 100%; }
  .downloads, .downloads .btn { width: 100%; }
  .actions .right { width: 100%; }
  .actions .right .btn { flex: 1; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .01ms !important; animation-delay: 0s !important; transition-duration: .01ms !important; }
}
</style>
</head>
<body>
<div class="shell">
  <header class="topbar">
    <a class="brand" href="/">
      <span class="brand-mark"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="9" cy="12" r="5"/><circle cx="15" cy="12" r="5"/></svg></span>
      EntityForge
    </a>
    <span class="engine" id="engine"></span>
  </header>

  <section class="intro">
    <h1 class="reveal">Resolve business entities across sources</h1>
    <p class="reveal d1">Pick one reference list, add the files you want to compare against it, and see which companies are the same, side by side.</p>
  </section>

  <section class="card reveal d2" aria-label="Matching workflow">
    <ol class="steps" id="steps">
      <li data-step="1"><span class="step-dot">1</span>Reference file</li>
      <li data-step="2"><span class="step-dot">2</span>Comparison files</li>
      <li data-step="3"><span class="step-dot">3</span>Results</li>
    </ol>

    <!-- Step 1: reference -->
    <div class="panel" id="panel-1">
      <div class="panel-head">
        <h2>Upload your reference file</h2>
        <p>The single source every other record is matched against. TSV or CSV with <code>entity_id</code> <code>business_name</code> <code>business_address</code> <code>country</code>.</p>
      </div>
      <label class="drop" id="ref-drop">
        <input id="ref-input" type="file" accept=".tsv,.csv">
        <span class="drop-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 15V4m0 0L8 8m4-4 4 4"/><path d="M4 14v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4"/></svg></span>
        <strong>Drop the reference file here</strong>
        <small>or click to browse · one file, up to 25 MB</small>
      </label>
      <ul class="file-list" id="ref-list"></ul>
      <div class="actions">
        <button class="btn btn-ghost" id="sample-btn" type="button">Try with sample data</button>
        <div class="right"><button class="btn btn-primary" id="to-step-2" type="button" disabled>Continue
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14m-5-5 5 5-5 5"/></svg></button></div>
      </div>
    </div>

    <!-- Step 2: comparison files -->
    <div class="panel" id="panel-2" hidden>
      <div class="ref-chip">
        <span class="tag">Reference</span>
        <span class="grow" id="ref-summary"></span>
        <button class="link-btn" type="button" data-go="1">Change</button>
      </div>
      <div class="panel-head">
        <h2>Add comparison files</h2>
        <p>One or more sources to match against the reference. Add as many as you need.</p>
      </div>
      <label class="drop" id="tgt-drop">
        <input id="tgt-input" type="file" accept=".tsv,.csv" multiple>
        <span class="drop-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 5v14M5 12h14"/></svg></span>
        <strong>Drop comparison files here</strong>
        <small>or click to browse · select several at once</small>
      </label>
      <ul class="file-list" id="tgt-list"></ul>
      <div id="run-error"></div>
      <div class="actions">
        <button class="btn btn-ghost" type="button" data-go="1">Back</button>
        <div class="right"><button class="btn btn-primary" id="run-btn" type="button" disabled>Run matching</button></div>
      </div>
    </div>

    <!-- Step 3a: running -->
    <div class="panel" id="panel-run" hidden>
      <div class="panel-head">
        <h2 id="run-title">Matching records…</h2>
        <p id="run-sub"></p>
      </div>
      <div class="run-state">
        <div class="progress"><i id="bar"></i></div>
        <ul class="stages" id="stages"></ul>
        <span class="elapsed" id="elapsed">0.0s</span>
      </div>
    </div>

    <!-- Step 3b: done -->
    <div class="panel" id="panel-done" hidden>
      <div class="done-state">
        <span class="done-badge"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="m5 12.5 4.5 4.5L19 7.5"/></svg></span>
        <div class="grow">
          <div class="panel-head" style="margin:0">
            <h2 id="done-title">Matching complete</h2>
            <p id="done-sub"></p>
          </div>
        </div>
        <button class="btn btn-ghost" type="button" id="new-run">Start a new run</button>
      </div>
    </div>
  </section>

  <section class="results" id="results" hidden aria-live="polite"></section>
</div>
<div class="toast" id="toast" role="status"></div>

<script>
const CONFIG = __CONFIG__;
const REQUIRED = ["entity_id", "business_name", "business_address", "country"];
const PAGE = 40;
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const plural = (n, one, many) => `${n.toLocaleString()} ${n === 1 ? one : (many || one + "s")}`;
const bytes = b => b < 1024 ? b + " B" : b < 1048576 ? (b / 1024).toFixed(1) + " KB" : (b / 1048576).toFixed(1) + " MB";
const ICON = {
  check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><path d="m5 12.5 4.5 4.5L19 7.5"/></svg>',
  file: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/></svg>',
  alert: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="9"/><path d="M12 7.5v5.5m0 3.2v.3"/></svg>',
  x: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 6l12 12M18 6 6 18"/></svg>',
  arrow: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14m-5-5 5 5-5 5"/></svg>',
  down: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 4v11m0 0-4-4m4 4 4-4M5 20h14"/></svg>',
  search: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="6.5"/><path d="m20 20-4-4"/></svg>',
};

$("engine").innerHTML = CONFIG.engineError ? "Matching engine unavailable" : `Model <b>${esc(CONFIG.model)}</b>`;
if (!CONFIG.sample) $("sample-btn").hidden = true;

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.classList.remove("show"), 3400);
}

/* ---------- Files: quick header check in the browser ---------- */
const state = { ref: null, targets: [], step: 1, running: false, result: null };

async function inspect(file) {
  const item = { file, name: file.name, size: file.size, rows: 0, error: null };
  if (!/\.(tsv|csv)$/i.test(file.name)) { item.error = "Only .tsv or .csv files are supported"; return item; }
  if (file.size > 25 * 1048576) { item.error = "Larger than 25 MB"; return item; }
  try {
    const text = (await file.text()).replace(/^﻿/, "");
    const lines = text.split(/\r?\n/).filter(line => line.trim());
    const sep = /\.tsv$/i.test(file.name) ? "\t" : ",";
    const header = (lines[0] || "").split(sep).map(h => h.trim().replace(/^"|"$/g, "").toLowerCase());
    const missing = REQUIRED.filter(col => !header.includes(col));
    item.rows = Math.max(0, lines.length - 1);
    if (missing.length) item.error = `Missing column${missing.length > 1 ? "s" : ""}: ${missing.join(", ")}`;
    else if (!item.rows) item.error = "No records in this file";
  } catch { item.error = "Couldn't read this file"; }
  return item;
}

function fileItem(item, index, removable) {
  const cls = item.error ? "bad" : "ok";
  const meta = item.error ? esc(item.error) : `${plural(item.rows, "record")} · ${bytes(item.size)}`;
  return `<li class="file ${cls}" style="animation-delay:${index * 40}ms">
    <span class="file-icon">${item.error ? ICON.alert : ICON.check}</span>
    <div style="min-width:0"><div class="file-name">${esc(item.name)}</div><div class="file-meta">${meta}</div></div>
    ${removable ? `<button class="icon-btn" type="button" data-remove="${index}" aria-label="Remove ${esc(item.name)}">${ICON.x}</button>` : "<span></span>"}
  </li>`;
}

function renderFiles() {
  $("ref-list").innerHTML = state.ref ? fileItem(state.ref, 0, false) : "";
  $("to-step-2").disabled = !state.ref || !!state.ref.error;
  $("tgt-list").innerHTML = state.targets.map((t, i) => fileItem(t, i, true)).join("");
  const valid = state.targets.filter(t => !t.error);
  $("run-btn").disabled = !valid.length || state.targets.some(t => t.error) || state.running;
  $("run-btn").textContent = valid.length ? `Match against ${plural(valid.length, "file")}` : "Run matching";
  if (state.ref && !state.ref.error) $("ref-summary").textContent = `${state.ref.name} · ${plural(state.ref.rows, "record")}`;
}

function setupDrop(dropId, inputId, onFiles) {
  const drop = $(dropId);
  ["dragenter", "dragover"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove("dragging"); }));
  drop.addEventListener("drop", e => onFiles(Array.from(e.dataTransfer.files)));
  $(inputId).addEventListener("change", e => { onFiles(Array.from(e.target.files)); e.target.value = ""; });
}

setupDrop("ref-drop", "ref-input", async files => {
  if (!files.length) return;
  if (files.length > 1) toast("The reference is a single file — using the first one.");
  state.ref = await inspect(files[0]);
  renderFiles();
});

setupDrop("tgt-drop", "tgt-input", async files => {
  const items = await Promise.all(files.map(inspect));
  items.forEach(item => {
    const existing = state.targets.findIndex(t => t.name === item.name);
    if (existing >= 0) state.targets[existing] = item; else state.targets.push(item);
  });
  setError("");
  renderFiles();
});

$("tgt-list").addEventListener("click", e => {
  const btn = e.target.closest("[data-remove]");
  if (!btn) return;
  btn.closest(".file").classList.add("leaving");
  setTimeout(() => { state.targets.splice(+btn.dataset.remove, 1); renderFiles(); }, 190);
});

function setError(message) {
  $("run-error").innerHTML = message ? `<div class="alert">${esc(message)}</div>` : "";
}

/* ---------- Steps ---------- */
const PANELS = { 1: "panel-1", 2: "panel-2", run: "panel-run", done: "panel-done" };
let visiblePanel = "panel-1", swapTimer = null;

function showPanel(key) {
  const next = PANELS[key];
  if (next === visiblePanel) return;
  clearTimeout(swapTimer);
  Object.values(PANELS).forEach(id => { if (id !== visiblePanel) { $(id).hidden = true; $(id).classList.remove("entering", "leaving"); } });
  const current = $(visiblePanel);
  current.classList.remove("entering");
  current.classList.add("leaving");
  swapTimer = setTimeout(() => {
    current.hidden = true;
    current.classList.remove("leaving");
    const panel = $(next);
    panel.hidden = false;
    panel.classList.remove("entering");
    void panel.offsetWidth;
    panel.classList.add("entering");
  }, 180);
  visiblePanel = next;
  const step = key === "run" || key === "done" ? 3 : +key;
  document.querySelectorAll("#steps li").forEach(li => {
    const n = +li.dataset.step;
    const done = n < step || (key === "done" && n === 3);
    li.classList.toggle("current", n === step && !done);
    li.classList.toggle("done", done);
    li.querySelector(".step-dot").innerHTML = done ? ICON.check : n;
  });
}
showPanel(1);
document.querySelectorAll("#steps li")[0].classList.add("current");

document.addEventListener("click", e => {
  const go = e.target.closest("[data-go]");
  if (go && !state.running) showPanel(go.dataset.go);
});
$("to-step-2").addEventListener("click", () => showPanel(2));
$("new-run").addEventListener("click", () => { state.ref = null; state.targets = []; setError(""); renderFiles(); showPanel(1); });

/* ---------- Running ---------- */
const STAGES = () => [
  ["Reading and validating files", 0],
  ["Running TF-IDF character n-gram blocking", 700],
  [`Scoring candidate pairs with ${CONFIG.model}`, 1600],
  ["Applying the F0.5 precision threshold", 3200],
];

async function run(request, label) {
  if (state.running) return;
  state.running = true;
  setError("");
  $("run-sub").textContent = label;
  const stages = STAGES();
  $("stages").innerHTML = stages.map(([text]) => `<li class="stage"><span class="stage-icon">${ICON.check}</span>${esc(text)}</li>`).join("");
  $("bar").style.width = "0%";
  showPanel("run");

  const started = performance.now();
  const items = [...$("stages").children];
  const tick = () => {
    const t = performance.now() - started;
    let active = 0;
    stages.forEach(([, at], i) => { if (t >= at) active = i; });
    items.forEach((li, i) => { li.classList.toggle("done", i < active); li.classList.toggle("active", i === active); });
    $("bar").style.width = (92 * (1 - Math.exp(-t / 2600))).toFixed(1) + "%";
    $("elapsed").textContent = (t / 1000).toFixed(1) + "s";
  };
  tick();
  const timer = setInterval(tick, 120);

  try {
    const res = await request();
    let data;
    try { data = await res.json(); } catch { throw new Error(`The server returned an unexpected response (${res.status}).`); }
    if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Matching failed.");
    clearInterval(timer);
    items.forEach(li => { li.classList.remove("active"); li.classList.add("done"); });
    $("bar").style.width = "100%";
    await new Promise(r => setTimeout(r, 450));
    state.result = data;
    const s = data.summary;
    $("done-title").textContent = `Matched ${plural(s.matched, "record")} of ${s.reference.records.toLocaleString()}`;
    $("done-sub").textContent = `${s.reference.name} against ${plural(s.sources.length, "file")} · ${plural(s.links, "link")} found in ${s.seconds}s`;
    showPanel("done");
    renderResults();
    setTimeout(() => $("results").scrollIntoView({ behavior: "smooth", block: "start" }), 250);
  } catch (err) {
    clearInterval(timer);
    showPanel(state.ref && state.targets.length ? 2 : 1);
    setTimeout(() => {
      if (visiblePanel === "panel-2") setError(err.message); else toast(err.message);
    }, 200);
  } finally {
    state.running = false;
    renderFiles();
  }
}

$("run-btn").addEventListener("click", () => {
  const form = new FormData();
  form.append("reference", state.ref.file);
  state.targets.forEach(t => form.append("targets", t.file));
  run(() => fetch("/api/resolve", { method: "POST", body: form }),
      `${state.ref.name} against ${plural(state.targets.length, "comparison file")}`);
});
$("sample-btn").addEventListener("click", () =>
  run(() => fetch("/api/resolve/sample", { method: "POST" }), "Sample data: test_source1.tsv against test_source2.tsv and test_source3.tsv"));

/* ---------- Results ---------- */
const view = { filter: "all", query: "", limit: PAGE };

// Count-up runs in CSS (@property), so the final number is always what's left on screen.
function statValue(n, suffix = "") {
  return n < 100000
    ? `<strong class="stat-value count" style="--to:${n}" data-suffix="${suffix}" aria-label="${n}${suffix}"></strong>`
    : `<strong class="stat-value">${n.toLocaleString()}${suffix}</strong>`;
}

function download(kind) {
  const names = { matching: "matching_results.tsv", candidates: "candidate_pairs.tsv" };
  const blob = new Blob([state.result.downloads[kind]], { type: "text/tab-separated-values" });
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement("a"), { href: url, download: names[kind] });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function renderResults() {
  const { summary: s, rows } = state.result;
  rows.forEach(r => {
    r._text = [r.id, r.name, r.address, r.country, ...r.matches.flatMap(m => [m.id, m.name, m.address, m.source])].join(" ").toLowerCase();
  });
  view.filter = "all"; view.query = ""; view.limit = PAGE;
  const pct = s.reference.records ? Math.round(s.matched / s.reference.records * 100) : 0;
  const maxLinks = Math.max(1, ...s.sources.map(x => x.links));
  const root = $("results");
  root.hidden = false;
  root.classList.remove("entering"); void root.offsetWidth; root.classList.add("entering");
  root.innerHTML = `
    <div class="results-head">
      <div>
        <h2>Results</h2>
        <p>${esc(s.reference.name)} · ${plural(s.reference.records, "reference record")} compared with ${plural(s.target_records, "record")}</p>
      </div>
      <div class="downloads">
        <button class="btn btn-ghost" type="button" data-download="candidates">${ICON.down}candidate_pairs.tsv</button>
        <button class="btn btn-primary" type="button" data-download="matching">${ICON.down}matching_results.tsv</button>
      </div>
    </div>
    <div class="stats">
      <div class="card stat"><span class="stat-label">Reference records</span>${statValue(s.reference.records)}<span class="stat-foot">${esc(s.reference.name)}</span></div>
      <div class="card stat" style="animation-delay:.06s"><span class="stat-label">Matched</span>${statValue(s.matched)}<span class="stat-foot">${pct}% of reference records</span></div>
      <div class="card stat" style="animation-delay:.12s"><span class="stat-label">No match</span>${statValue(s.unmatched)}<span class="stat-foot">No confident link found</span></div>
      <div class="card stat" style="animation-delay:.18s"><span class="stat-label">Avg. match score</span>${statValue(s.avg_score, "%")}<span class="stat-foot">${plural(s.links, "link")} · model ${esc(s.model)}</span></div>
    </div>
    <div class="card sources">
      <h3>Links by comparison file</h3>
      ${s.sources.map(x => `<div class="source-row"><span class="name" title="${esc(x.name)}">${esc(x.name)}</span><span class="track"><i data-w="${Math.round(x.links / maxLinks * 100)}"></i></span><span class="num">${plural(x.links, "link")} · ${x.records.toLocaleString()} records</span></div>`).join("")}
    </div>
    <div class="toolbar">
      <div class="seg" id="seg"><span class="thumb"></span>
        <button type="button" data-filter="all" class="active">All</button>
        <button type="button" data-filter="matched">Matched</button>
        <button type="button" data-filter="none">No match</button>
      </div>
      <label class="search">${ICON.search}<span class="sr-only">Search</span><input id="q" type="search" placeholder="Search company, address, ID…" autocomplete="off"></label>
    </div>
    <div class="pairs" id="pairs"></div>
    <div id="more"></div>`;

  requestAnimationFrame(() => {
    root.querySelectorAll("[data-w]").forEach(el => { el.style.width = el.dataset.w + "%"; });
    moveThumb();
  });
  root.querySelectorAll("[data-download]").forEach(b => b.addEventListener("click", () => download(b.dataset.download)));
  $("seg").addEventListener("click", e => {
    const b = e.target.closest("[data-filter]");
    if (!b) return;
    view.filter = b.dataset.filter; view.limit = PAGE;
    moveThumb(); renderPairs();
  });
  $("q").addEventListener("input", e => { view.query = e.target.value.trim().toLowerCase(); view.limit = PAGE; renderPairs(); });
  renderPairs();
}

function moveThumb() {
  const seg = $("seg");
  if (!seg) return;
  seg.querySelectorAll("button").forEach(b => b.classList.toggle("active", b.dataset.filter === view.filter));
  const active = seg.querySelector("button.active"), thumb = seg.querySelector(".thumb");
  thumb.style.width = active.offsetWidth + "px";
  thumb.style.transform = `translateX(${active.offsetLeft}px)`;
}

function entityCard(e, label, badge) {
  return `<div class="entity">
    <div class="entity-top"><span class="label" title="${esc(label)}">${esc(label)}</span><span class="mono">${esc(e.id)}</span></div>
    <div class="entity-name">${esc(e.name)}</div>
    <div class="entity-addr">${esc(e.address)}</div>
    <div class="entity-foot">${e.country ? `<span class="country">${esc(e.country)}</span>` : "<span></span>"}${badge || ""}</div>
  </div>`;
}

function renderPairs() {
  const rows = state.result.rows.filter(r =>
    (view.filter === "all" || (view.filter === "matched") === (r.matches.length > 0)) &&
    (!view.query || r._text.includes(view.query)));
  const shown = rows.slice(0, view.limit);
  $("pairs").innerHTML = shown.length ? shown.map((r, i) => {
    const right = r.matches.length
      ? r.matches.map(m => entityCard(m, m.source, `<span class="score match">${m.score}% match</span>`)).join("")
      : `<div class="entity empty"><div class="no-match">No confident match in the comparison files.</div>
          ${r.nearest ? `<div class="nearest">Closest candidate: <b>${esc(r.nearest.name)}</b> · ${esc(r.nearest.source)} · ${r.nearest.score}% score</div>` : ""}</div>`;
    return `<article class="card pair ${r.matches.length ? "matched" : ""}" style="animation-delay:${Math.min(i, 10) * 35}ms">
      ${entityCard(r, "Reference")}
      <span class="connector">${ICON.arrow}</span>
      <div class="stack">${right}</div>
    </article>`;
  }).join("") : `<div class="card list-empty">No records match this search.</div>`;
  const rest = rows.length - shown.length;
  $("more").innerHTML = rest > 0 ? `<div class="more"><button class="btn btn-ghost" type="button" id="more-btn">Show ${Math.min(rest, PAGE)} more · ${rest.toLocaleString()} left</button></div>` : "";
  if (rest > 0) $("more-btn").addEventListener("click", () => { view.limit += PAGE; renderPairs(); });
}

window.addEventListener("resize", moveThumb);
renderFiles();
if (CONFIG.engineError) toast("The matching engine couldn't start on this server. Check the deployment logs.");
</script>
</body>
</html>
"""
