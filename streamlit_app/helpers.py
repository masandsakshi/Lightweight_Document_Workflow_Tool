import requests
import json
from typing import List, Dict, Any

API_BASE = "http://localhost:8000"

def upload_pdf(file_bytes, filename: str) -> Dict[str, Any]:
	files = {"file": (filename, file_bytes, "application/pdf")}
	r = requests.post(f"{API_BASE}/upload", files=files, timeout=60)
	r.raise_for_status()
	return r.json()

def extract(file_id: str, filename: str, ocr_on_all: bool=False) -> Dict[str, Any]:
	params = {"file_id": file_id, "filename": filename, "ocr_on_all": str(ocr_on_all).lower()}
	r = requests.get(f"{API_BASE}/extract", params=params, timeout=120)
	r.raise_for_status()
	return r.json()

def download(file_id: str, filename: str, fmt: str="json") -> bytes:
	params = {"file_id": file_id, "filename": filename, "format": fmt}
	r = requests.get(f"{API_BASE}/download", params=params, timeout=60)
	r.raise_for_status()
	return r.content

def pretty_json(data: Any) -> str:
	return json.dumps(data, indent=2, ensure_ascii=False)
