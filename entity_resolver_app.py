"""FastAPI dashboard for the business entity-resolution pipeline."""
from __future__ import annotations

import html
import json
import subprocess
import sys
from threading import Lock
from difflib import SequenceMatcher
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "dataset" / "test"
OUTPUT_DIR = BASE_DIR / "output"
PIPELINE = BASE_DIR / "run_solution.py"
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


def run_pipeline() -> str:
    result = subprocess.run(
        [sys.executable, str(PIPELINE), "--train-dir", str(BASE_DIR / "dataset" / "train"),
         "--test-dir", str(TEST_DIR), "--output-dir", str(OUTPUT_DIR)],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout or "Pipeline failed").strip()[-3000:]
        raise HTTPException(status_code=500, detail=detail)
    return (result.stdout or "Pipeline completed").strip()


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
    rows_json = (json.dumps(rows, ensure_ascii=False)
                 .replace("<", "\\u003c")
                 .replace(">", "\\u003e")
                 .replace("&", "\\u0026"))
    notice = f'<div class="notice">{html.escape(message)}</div>' if message else ""
    template = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>EntityForge BER — Entity Intelligence</title><script src="https://cdn.jsdelivr.net/npm/apexcharts"></script><script>window.ApexCharts=window.ApexCharts||class{constructor(element){this.element=element}render(){this.element.innerHTML='<div class="empty">Chart library unavailable — metrics remain available.</div>';return Promise.resolve()}};</script>
<style>
.drop-size{display:inline-block;color:#94a3b8;font-size:11px;margin-top:8px}
.grid{display:grid;grid-template-columns:1.2fr .8fr;gap:18px;margin-bottom:18px}.panel{padding:22px}.panel-title{display:flex;justify-content:space-between;align-items:center;gap:14px;margin-bottom:18px}.panel-title h2{font-size:16px;margin:0}.panel-title span{color:var(--muted);font-size:12px}.chart{min-height:220px}.upload{border-style:dashed}.drop-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.drop{position:relative;border:1px dashed #475569;border-radius:13px;padding:16px;transition:.25s background,.25s border-color;background:#111827}.drop:hover,.drop.ready{border-color:#64748b;background:#172235}.drop input{position:absolute;inset:0;opacity:0;cursor:pointer}.drop-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:22px}.source-tag{font-size:10px;color:#a6aec0;text-transform:uppercase;letter-spacing:.12em;font-weight:800}.check{display:none;color:var(--green);font-size:18px}.drop.ready .check{display:block}.drop-name{font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.drop-hint{color:var(--muted);font-size:12px;margin-top:5px}.run-row{display:flex;justify-content:space-between;align-items:center;gap:18px;margin-top:18px}.run-row small{color:var(--muted)}.progress{height:3px;flex:1;background:#202939;border-radius:9px;overflow:hidden;display:none}.progress i{display:block;height:100%;width:40%;background:#4f46e5;animation:load 1s infinite}@keyframes load{to{transform:translateX(260%)}}
.toolbar{display:flex;justify-content:space-between;gap:16px;align-items:center;margin-bottom:16px}.toolbar h2{margin:0;font-size:17px}.controls{display:flex;gap:8px;align-items:center}.search{width:min(310px,40vw);background:#0b1018;border:1px solid var(--line);border-radius:9px;padding:10px 12px;color:var(--text);outline:none}.search:focus{border-color:#64748b}.toggle{display:flex;padding:3px;border:1px solid var(--line);border-radius:9px;background:#0a0e15}.toggle button{background:transparent;border:0;color:var(--muted);padding:7px 9px;font-size:12px}.toggle button.active{background:#1e293b;color:#cbd5e1;box-shadow:none;transform:none}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:11px}table{width:100%;border-collapse:collapse;min-width:920px}th,td{padding:14px 15px;text-align:left;border-bottom:1px solid #1d2531;vertical-align:top}th{background:#0c1119;color:#687589;text-transform:uppercase;font-size:10px;letter-spacing:.1em}tr{animation:rise .3s ease both}tr:hover td{background:#151b27}tr:last-child td{border:0}.id{font-weight:750;color:#dce2ed;white-space:nowrap}.company{font-weight:700}.muted{color:var(--muted);font-size:12px;line-height:1.55}.pill{display:inline-flex;align-items:center;gap:5px;background:#1e293b;border:1px solid #334155;border-radius:99px;color:#cbd5e1;padding:5px 8px;margin:2px;font-size:11px}.match{margin-bottom:8px}.match:last-child{margin:0}.confidence{display:inline-block;color:#86cdb0;background:#12352c;border:1px solid #27634e;border-radius:99px;padding:4px 7px;font-size:10px;font-weight:800;margin-left:5px}.empty{color:#6c7789}.cards{display:none;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}.entity-card{padding:18px;border:1px solid var(--line);border-radius:13px;background:#0d131d}.entity-card .arrow{color:#94a3b8;font-size:18px;margin:10px 0}.hidden{display:none!important}@keyframes rise{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}
@media(max-width:960px){.metrics{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}.hero{display:block}.hero .actions{margin-top:24px}}@media(max-width:640px){.shell{padding:20px 15px 50px}.nav{margin-bottom:48px}.status{font-size:11px}.drop-grid{grid-template-columns:1fr}.metrics{gap:9px}.metric{padding:15px}.metric-value{font-size:26px}.toolbar{display:block}.controls{margin-top:14px}.search{width:100%}.toggle{margin-top:8px;width:max-content}}
</style></head><body><main class="shell">
<nav class="nav"><div class="brand"><span class="logo"><svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.3"><path d="M12 3v18M3 12h18"/></svg></span>EntityForge<span style="color:#778294;font-weight:500">/ BER</span></div><div class="status"><i class="dot"></i> ML Model Pipeline Ready</div></nav>
<section class="hero"><div><div class="eyebrow">Entity intelligence · v1.0</div><h1>Make every match<br><span style="color:#cbd5e1">feel certain.</span></h1><p>Precision-first entity resolution for fragmented business data. Upload your sources, run the model, and explore every relationship.</p></div><div class="actions"><a class="button ghost" href="/download/candidates">↓ Candidates</a><a class="button" href="/download/matching">↓ Export results</a></div></section>
__NOTICE__<section class="metrics"><div class="metric"><span class="metric-label">Total S1 entities</span><strong class="metric-value">__TOTAL__</strong><span class="metric-foot">Reference records processed</span></div><div class="metric"><span class="metric-label">Matched pairs</span><strong class="metric-value">__MATCHED__</strong><span class="metric-foot"><strong>● Active</strong> precision pipeline</span></div><div class="metric"><span class="metric-label">Singletons</span><strong class="metric-value">__SINGLETONS__</strong><span class="metric-foot">No confident relationship</span></div><div class="metric"><span class="metric-label">Mean confidence</span><strong class="metric-value">__CONFIDENCE__%</strong><span class="metric-foot">Across all predicted links</span></div></section>
<div class="grid"><section class="panel"><div class="panel-title"><h2>Confidence distribution</h2><span>Similarity score · live snapshot</span></div><div id="confidence-chart" class="chart"></div></section><section class="panel"><div class="panel-title"><h2>Match source split</h2><span>Linked entities</span></div><div id="source-chart" class="chart"></div></section></div>
<section class="panel upload"><div class="panel-title"><div><h2>Run resolution pipeline</h2><span>Drop your three tab-separated source files to begin</span></div><span class="eyebrow">INPUT HUB</span></div><form id="upload-form" action="/upload-and-run/" method="post" enctype="multipart/form-data"><div class="drop-grid"><label class="drop" id="drop-source1"><input name="source1" type="file" accept=".tsv" required><div class="drop-top"><span class="source-tag">Source 01 · reference</span><span class="check">✓</span></div><div class="drop-name">Choose test_source1.tsv</div><div class="drop-hint">TSV · business entities</div><span class="drop-size">No file selected</span></label><label class="drop" id="drop-source2"><input name="source2" type="file" accept=".tsv" required><div class="drop-top"><span class="source-tag">Source 02</span><span class="check">✓</span></div><div class="drop-name">Choose test_source2.tsv</div><div class="drop-hint">TSV · fragmented records</div><span class="drop-size">No file selected</span></label><label class="drop" id="drop-source3"><input name="source3" type="file" accept=".tsv" required><div class="drop-top"><span class="source-tag">Source 03</span><span class="check">✓</span></div><div class="drop-name">Choose test_source3.tsv</div><div class="drop-hint">TSV · fragmented records</div><span class="drop-size">No file selected</span></label></div><div class="run-row"><small id="run-status">Ready when you are.</small><div class="progress" id="progress"><i></i></div><button type="submit">Run resolution pipeline <span>↗</span></button></div></form></section>
<section class="panel"><div class="toolbar"><h2>Entity matches <span class="muted">· __TOTAL__ records</span></h2><div class="controls"><input id="search" class="search" type="search" placeholder="⌕  Search name, ID, country..."><div class="toggle"><button class="active" data-view="table">Table</button><button data-view="cards">Cards</button></div></div></div><div id="table-view" class="table-wrap"><table><thead><tr><th>Reference entity</th><th>Company profile</th><th>Resolved links</th><th>Model signal</th></tr></thead><tbody id="results"></tbody></table></div><div id="cards-view" class="cards"></div></section>
</main><script>const rows=__ROWS_JSON__;const esc=s=>String(s??'').replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[c]));const confidence=rows.flatMap(r=>r.matched.map(x=>x.confidence));
function matchMarkup(r){return r.matched.length?r.matched.map(x=>`<div class="match"><span class="pill">${esc(x.source)} · ${esc(x.entity_id)}</span><span class="confidence">${esc(x.confidence)}%</span><br><span class="muted">${esc(x.business_name)}<br>${esc(x.business_address)}</span></div>`).join(''):'<span class="empty">No confident match</span>'}function draw(){const q=document.getElementById('search').value.toLowerCase();const visible=rows.filter(r=>JSON.stringify(r).toLowerCase().includes(q));document.getElementById('results').innerHTML=visible.map(r=>`<tr><td class="id">${esc(r.source1.entity_id)}</td><td><div class="company">${esc(r.source1.business_name)}</div><span class="muted">${esc(r.source1.business_address)}<br>${esc(r.source1.country)}</span></td><td>${matchMarkup(r)}</td><td><span class="confidence">${r.matched.length?Math.round(r.matched.reduce((a,x)=>a+x.confidence,0)/r.matched.length):0}% precision</span><br><span class="muted">${r.candidates.length} candidates ranked</span></td></tr>`).join('')||'<tr><td colspan="4" class="empty">No records found.</td></tr>';document.getElementById('cards-view').innerHTML=visible.map(r=>`<article class="entity-card"><span class="id">${esc(r.source1.entity_id)}</span><div class="company" style="margin-top:10px">${esc(r.source1.business_name)}</div><span class="muted">${esc(r.source1.business_address)}<br>${esc(r.source1.country)} · ${r.candidates.length} candidates</span><div class="arrow">↓</div>${matchMarkup(r)}<span class="muted">Model signal: ${r.matched.length?Math.round(r.matched.reduce((a,x)=>a+x.confidence,0)/r.matched.length):0}% precision</span></article>`).join('')||'<div class="empty">No records found.</div>'}document.getElementById('search').addEventListener('input',draw);document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('[data-view]').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.getElementById('table-view').classList.toggle('hidden',b.dataset.view!=='table');document.getElementById('cards-view').style.display=b.dataset.view==='cards'?'grid':'none'}));draw();
new ApexCharts(document.querySelector('#confidence-chart'),{series:[{name:'Matches',data:[confidence.filter(x=>x<60).length,confidence.filter(x=>x>=60&&x<75).length,confidence.filter(x=>x>=75&&x<90).length,confidence.filter(x=>x>=90).length]}],chart:{type:'bar',height:220,toolbar:{show:false},background:'transparent'},plotOptions:{bar:{borderRadius:5,columnWidth:'45%'}},colors:['#4f46e5'],dataLabels:{enabled:false},grid:{borderColor:'#334155',strokeDashArray:4},xaxis:{categories:['<60','60–75','75–90','90–100'],labels:{style:{colors:'#94a3b8'}},axisBorder:{show:false},axisTicks:{show:false}},yaxis:{labels:{style:{colors:'#94a3b8'}}},tooltip:{theme:'dark'}}).render();new ApexCharts(document.querySelector('#source-chart'),{series:[__S2__,__S3__],labels:['Source 2','Source 3'],chart:{type:'donut',height:220,background:'transparent'},colors:['#4f46e5','#059669'],stroke:{colors:['#10151f']},legend:{position:'bottom',labels:{colors:'#94a3b8'}},dataLabels:{enabled:false},plotOptions:{pie:{donut:{size:'72%',labels:{show:true,total:{show:true,label:'LINKS',color:'#94a3b8',fontSize:'11px'},value:{color:'#e2e8f0',fontSize:'24px',fontWeight:700}}}}},tooltip:{theme:'dark'}}).render();
function sizeLabel(bytes){if(bytes<1024)return bytes+' B';if(bytes<1048576)return (bytes/1024).toFixed(1)+' KB';return (bytes/1048576).toFixed(1)+' MB'}document.querySelectorAll('.drop input').forEach(input=>input.addEventListener('change',()=>{const drop=input.closest('.drop');drop.classList.toggle('ready',!!input.files.length);if(input.files.length){drop.querySelector('.drop-name').textContent=input.files[0].name;drop.querySelector('.drop-size').textContent=sizeLabel(input.files[0].size)+' · TSV'}}));document.getElementById('upload-form').addEventListener('submit',()=>{document.getElementById('run-status').textContent='Processing entity matches…';document.getElementById('progress').style.display='block'});</script></body></html>"""
    return (template.replace("__NOTICE__", notice).replace("__TOTAL__", str(summary["total"])).replace("__MATCHED__", str(summary["matched"]))
            .replace("__SINGLETONS__", str(summary["singletons"])).replace("__CONFIDENCE__", str(summary["mean_confidence"]))
            .replace("__S2__", str(summary["source2_matches"])).replace("__S3__", str(summary["source3_matches"])).replace("__ROWS_JSON__", rows_json))


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return render_dashboard()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def snapshot_files(paths: list[Path]) -> dict[Path, bytes | None]:
    return {path: path.read_bytes() if path.exists() else None for path in paths}


def restore_files(snapshot: dict[Path, bytes | None]) -> None:
    for path, content in snapshot.items():
        if content is None:
            if path.exists():
                path.unlink()
        else:
            path.write_bytes(content)


@app.post("/process", response_class=HTMLResponse)
@app.post("/upload-and-run/", response_class=HTMLResponse, include_in_schema=False)
async def process(
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
                temporary = destination.with_suffix(destination.suffix + ".uploading")
                temporary.write_bytes(data)
                temporary.replace(destination)
            pipeline_output = run_pipeline()
        except Exception:
            restore_files(snapshot)
            raise
    log = pipeline_output.splitlines()[-1] if pipeline_output else "Pipeline completed"
    return render_dashboard(f"Matching job completed: {log}")


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
