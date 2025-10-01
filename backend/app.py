from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Optional
import uuid
import shutil
import pandas as pd
try:
	from .extraction import extract_table_from_pdf  
except ImportError:
	from extraction import extract_table_from_pdf  

STORAGE_DIR = Path("uploaded_pdfs")
OUTPUT_DIR = Path("api_output")
STORAGE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="PDF Extraction API", version="0.1.0")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_methods=["*"],
	allow_headers=["*"],
	allow_credentials=True,
)

def _save_df(df: pd.DataFrame, stem: str):
	csv_path = OUTPUT_DIR / f"{stem}.csv"
	json_path = OUTPUT_DIR / f"{stem}.json"
	df.to_csv(csv_path, index=False)
	df.to_json(json_path, orient="records", indent=2)
	return csv_path, json_path

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
	if not file.filename.lower().endswith('.pdf'):
		raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
	file_id = uuid.uuid4().hex
	target_path = STORAGE_DIR / f"{file_id}_{file.filename}"
	with target_path.open('wb') as buffer:
		shutil.copyfileobj(file.file, buffer)
	return {"file_id": file_id, "filename": file.filename}

@app.get("/extract")
def extract(file_id: str = Query(...), filename: str = Query(...), ocr_on_all: bool = Query(False)):
	pattern = f"{file_id}_{filename}"
	pdf_path = STORAGE_DIR / pattern
	if not pdf_path.exists():
		raise HTTPException(status_code=404, detail="File not found. Upload first.")
	df = extract_table_from_pdf(str(pdf_path), ocr_on_all=ocr_on_all, debug=False)
	if df.empty:
		return {"data": [], "message": "No data extracted."}
	stem = pdf_path.stem
	csv_path, json_path = _save_df(df, stem)
	return {"data": df.to_dict(orient='records'), "csv": str(csv_path), "json": str(json_path)}

@app.get("/download")
def download(file_id: str = Query(...), filename: str = Query(...), format: str = Query("json")):
	stem = f"{file_id}_{filename}".replace('.pdf', '')
	target = OUTPUT_DIR / f"{stem}.{ 'json' if format.lower()=='json' else 'csv'}"
	if not target.exists():
		raise HTTPException(status_code=404, detail="Requested output not found. Run /extract first.")
	media_type = 'application/json' if target.suffix == '.json' else 'text/csv'
	return FileResponse(path=target, media_type=media_type, filename=target.name)

@app.get("/health")
def health():
	return {"status": "ok"}

