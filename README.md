# Lightweight Document Workflow Tool

Extract semi‑structured financial / remittance style data from PDFs and quickly correct, validate, and export it. The stack combines a heuristic extraction pipeline (machine text + OCR fallback + lightweight NER), a FastAPI backend service, and a Streamlit frontend for interactive review and editing.

---
## Key Capabilities
| Area | Description |
|------|-------------|
| PDF Ingestion | Hybrid: machine text via `pdfplumber`, OCR fallback via `pytesseract` |
| Heuristic Extraction | Regex + fuzzy keyword matching using Levenshtein distance + context keyword semantic matching |
| Line Reconstruction | Spatial clustering of word boxes into logical lines before entity extraction utilising layout parsing. |
| API Service | Endpoints for upload, extract, download |
| Frontend UI | Upload → extract → view → edit (JSON or csv) → preview → download corrected outputs. |
| Edit & Update capability | Independent JSON text editing OR csv editing; final corrected dataset downloadable separately. |
| MultiFormat Output | Tabulated data outputs in 2 formats: CSV and Json |
---
## Repository Structure
```
ReadMe.md               
requirements.txt         
backend/
  app.py                 
  extraction.py          
streamlit_app/
  app.py                 
  helpers.py             
uploaded_pdfs/          
api_output/              
Sample Documents/        
```

---
## Extraction Pipeline Overview
1. PDF Page Iteration: Each page processed through machine text extraction first.
2. OCR Fallback:  If the pdf was not machine readable, utilise pytesseract for OCR fallback text extraction.
3. Box Normalization: utilising to_data function get - (text, x, y, width, height, confidence, page).
4. Line Clustering: Y-axis based clustering with → left-to-right ordering. [Assuming straight line, everything in the same Y axis is in the same line]
5. Token Classification based on Named Entity Recognition:
	- Regex (high precision): account numbers, SWIFT, dates, currency, amount. --> Used for Numeric Data extraction.
	- Fuzzy keyword matching (Levenshtein similarity) on vocabulary phrases.
	- Contextual semantic matching (keywords within the full line context).
6. Span Merging: Consecutive tokens with same label will be merged.
7. Field Assembly: Longest / currency-containing / first-occurrence heuristics per field.
8. Cleaning & Normalization: Collapse whitespace, strip noise, format amount & codes.
9. Output: Single-row DataFrame (or empty if nothing extracted) → persisted as CSV & JSON.

---
## FastAPI Backend

### Endpoints
| Method | Path | Query / Body | Description |
|--------|------|--------------|-------------|
| POST | `/upload` | multipart/form-data (pdf) | Store PDF; returns `file_id` + original filename. |
| GET | `/extract` | `file_id`, `filename`, `ocr_on_all` (bool) | Run pipeline; returns records + saved output paths. |
| GET | `/download` | `file_id`, `filename`, `format` (`json|csv`) | Download previously extracted file. |

### Workflow
1. Upload PDF (sidebar).
2. Click “Run Extraction”.
3. Inspect original JSON + tabular view (left column).
4. Make corrections using:
	- JSON Editor (free-form list[dict]) OR
	- CSV (data editor) tab.
5. Preview & download corrected vs original outputs.

---
## Installation & Setup (Windows Focus)
1. Preferred: Create a virtual environment:
```powershell
python -m venv myenv
myenv\Scripts\Activate.ps1
```
2. Install backend dependencies:
```powershell
pip install -r requirements.txt
```
3. Install Tesseract OCR:
	- Download: https://github.com/tesseract-ocr/tesseract
	- Default path assumed in code: `C:/Program Files/Tesseract-OCR/tesseract.exe`
	- Please change path according to your installation.

---
## Running the System
Open two terminals (PowerShell):

Backend (FastAPI + auto-reload):
```powershell
uvicorn backend.app:app --reload --port 8000
```

Frontend (Streamlit UI):
```powershell
streamlit run streamlit_app\app.py --server.port 8501
```

Visit: http://localhost:8501

