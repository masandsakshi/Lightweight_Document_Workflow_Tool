# This file will extract the text from the pdfs and return csv and json files 
# for tabulated invoice and remittation data.

# IMPORTS
import pdfplumber
import pytesseract
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

#Please change the path according to your tesseract installation
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# STEP 1: Extract text boxes from PDF : first we try to check for machin readable text then use OCR as a fallback
def extract_text_boxes(file_path: str, ocr_on_all: bool = False,
                       ocr_conf_min: int = 25, dpi: int = 300,
                       debug: bool = False) -> List[Dict]:
    boxes: List[Dict] = []
    with pdfplumber.open(file_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            added_any = False

            # Native (machine-readable) extraction
            if not ocr_on_all:
                try:
                    words = page.extract_words() or []
                    for w in words:
                        if not w.get('text'):
                            continue
                        boxes.append({
                            'text': w['text'],
                            'x': int(w['x0']),
                            'y': int(w['top']),
                            'width': int(w['x1'] - w['x0']),
                            'height': int(w['bottom'] - w['top']),
                            'conf': 99,
                            'page': page_idx
                        })
                        added_any = True
                except Exception as e:
                    if debug:
                        print(f"[DEBUG] extract_words failed p{page_idx+1}: {e}")

            # OCR fallback (or forced)
            if ocr_on_all or not added_any:
                if debug:
                    print(f"[DEBUG] Page {page_idx+1}: OCR pass (added_any={added_any})")
                image = page.to_image(resolution=dpi)
                ocr = pytesseract.image_to_data(image.original, lang='eng', output_type=pytesseract.Output.DICT)
                for i, raw_txt in enumerate(ocr['text']):
                    txt = (raw_txt or "").strip()
                    conf_raw = ocr['conf'][i]
                    try:
                        conf = int(conf_raw)
                    except (ValueError, TypeError):
                        continue
                    if conf >= ocr_conf_min and txt:
                        boxes.append({
                            'text': txt,
                            'x': ocr['left'][i],
                            'y': ocr['top'][i],
                            'width': ocr['width'][i],
                            'height': ocr['height'][i],
                            'conf': conf,
                            'page': page_idx
                        })
    if debug:
        print(f"[DEBUG] Total boxes: {len(boxes)}")
    return boxes

# STEP 2: Cluster text boxes into lines based on y-coordinate proximity -- Assuming that lines are mostly horizontal and whatever is on the same y axis will be in the same line
def cluster_into_lines(boxes: List[Dict], y_threshold: int = 8) -> List[List[Dict]]:
    if not boxes:
        return []
    # Sort stable by page then y then x
    boxes_sorted = sorted(boxes, key=lambda b: (b['page'], b['y'], b['x']))
    lines: List[List[Dict]] = []
    current: List[Dict] = [boxes_sorted[0]]

    def line_baseline(line: List[Dict]) -> float:
        return sum(b['y'] for b in line) / len(line)

    for b in boxes_sorted[1:]:
        if b['page'] != current[-1]['page']:
            lines.append(sorted(current, key=lambda x: x['x']))
            current = [b]
            continue
        baseline = line_baseline(current)
        if abs(b['y'] - baseline) <= y_threshold:
            current.append(b)
        else:
            lines.append(sorted(current, key=lambda x: x['x']))
            current = [b]
    if current:
        lines.append(sorted(current, key=lambda x: x['x']))
    return lines


# Step 3: Named Entity Recognition (regex + fuzzy matching + semantic) ----------------
# We emulate a lightweight spaCy-like doc.ents pipeline.

# Reliable regex patterns for ONLY specific entity types.
REGEX_PATTERNS: Dict[str, re.Pattern] = {
    'beneficiary_account': re.compile(r'\b\d{2,3}[-\s]?\d{6,}[-\s]?\d{0,4}\b'),
    'swift_code': re.compile(r'\b[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?\b'),
    'date': re.compile(r'\b\d{1,2}[-/](?:\d{1,2}|[A-Za-z]{3})[-/]\d{2,4}\b'),
    'amount': re.compile(r'\b(?:USD|HKD)\s?\d[\d,]*(?:\.\d{2})?\b', re.IGNORECASE)
}
# Levenshtein -- implemented using DP (for fuzzy label matching) ----------------
def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            cost = 0 if ca.lower() == cb.lower() else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost     # substitution
            )
            prev = cur
    return dp[-1]

def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    dist = levenshtein_distance(a, b)
    return 1 - dist / max(len(a), len(b))

# VOCAB
FIELD_PATTERNS: Dict[str, List[str]] = {
    'beneficiary_name': ['beneficiary', 'name of beneficiary', 'account name'],
    'beneficiary_address': ['beneficiary address', 'address of beneficiary'],
    'beneficiary_account': ['account no', 'a/c no', 'account number', 'account'],
    'bank_name': ['bank name', 'correspondent bank', 'bank'],
    'swift_code': ['swift', 'bic', 'swift code'],
    'amount': ['amount', 'currency and amount', 'usd', 'hkd'],
    'applicant': ['remitter', 'applicant', 'sender'],
    'date': ['date'],
}

FUZZY_LABELS = FIELD_PATTERNS

CONTEXT_KEYWORDS = {
    'amount': ['usd', 'hkd', 'amount', 'currency'],
    'beneficiary_account': ['account', 'acct', 'a/c'],
    'swift_code': ['swift', 'bic'],
    'beneficiary_name': ['beneficiary'],
    'beneficiary_address': ['address'],
    'bank_name': ['bank', 'correspondent'],
    'applicant': ['remitter', 'applicant', 'sender'],
    'date': ['date']
}

MULTILINE_FIELDS = {'beneficiary_address', 'beneficiary_name'}

def _fuzzy_best_label(token: str) -> Tuple[Optional[str], float, str]:
    """Return (label, score, matched_pattern)."""
    if not token:
        return None, 0.0, ''
    best_label = None
    best_score = 0.0
    best_pat = ''
    for label, phrases in FUZZY_LABELS.items():
        for phrase in phrases:
            s = similarity(token, phrase)
            # adaptive thresholding: very short tokens require higher similarity
            min_needed = 0.9 if len(token) <= 4 else 0.78
            if s >= min_needed and s > best_score:
                best_label, best_score, best_pat = label, s, phrase
    return best_label, best_score, best_pat

def _regex_label(token: str) -> Optional[str]:
    raw = token.strip()
    if not raw:
        return None
    for label, pattern in REGEX_PATTERNS.items():
        if pattern.search(raw.upper() if label == 'swift_code' else raw):
            return label
    return None

def _context_label(token: str, line_text: str) -> Optional[str]:
    lt = line_text.lower()
    tok_norm = token.lower()
    if any(k in lt for k in CONTEXT_KEYWORDS['amount']) and re.search(r'\d', tok_norm):
        if re.search(r'(usd|hkd)', tok_norm) or re.search(r'\d', tok_norm):
            return 'amount'
    if 'beneficiary' in lt and not re.search(r'\d', tok_norm):
        return 'beneficiary_name'
    if 'address' in lt:
        return 'beneficiary_address'
    if any(k in lt for k in CONTEXT_KEYWORDS['applicant']):
        return 'applicant'
    if any(k in lt for k in CONTEXT_KEYWORDS['bank_name']):
        return 'bank_name'
    if any(k in lt for k in CONTEXT_KEYWORDS['beneficiary_account']) and re.search(r'\d', tok_norm):
        return 'beneficiary_account'
    if any(k in lt for k in CONTEXT_KEYWORDS['swift_code']) and (len(tok_norm) in (8, 11)) and tok_norm.isalnum():
        return 'swift_code'
    if 'date' in lt and re.search(r'\d', tok_norm):
        return 'date'
    return None

def classify_token(token: str, line_text: str) -> Tuple[Optional[str], float, str]:
    """Return (label, confidence, method)."""
    rlabel = _regex_label(token)
    if rlabel:
        return rlabel, 1.0, 'regex'
    flabel, fscore, pat = _fuzzy_best_label(token)
    if flabel:
        return flabel, fscore, f'fuzzy:{pat}'
    clabel = _context_label(token, line_text)
    if clabel:
        return clabel, 0.6, 'context'
    return None, 0.0, ''

def ner_classify(lines: List[List[Dict]], debug: bool = False) -> List[Dict]:
    """
    Produce a list of entity dicts similar in spirit to spaCy's doc.ents:
      { 'text': str, 'label': label, 'line': line_index, 'tokens': [token_texts] }
    Contiguous tokens with same label are merged.
    """
    entities: List[Dict] = []
    for idx, line_boxes in enumerate(lines):
        if not line_boxes:
            continue
        line_text = ' '.join(b['text'] for b in line_boxes)
        current = None
        for b in line_boxes:
            tok = b['text']
            label, score, method = classify_token(tok, line_text)
            if label is None:
                if current is not None:
                    entities.append(current)
                    current = None
                continue
            if current and current['label'] == label:
                current['text'] += ' ' + tok
                current['tokens'].append(tok)
            else:
                if current:
                    entities.append(current)
                current = {
                    'text': tok,
                    'label': label,
                    'line': idx,
                    'tokens': [tok],
                    'method': method
                }
        if current:
            entities.append(current)
    if debug:
        for e in entities:
            print(f"[NER] line={e['line']} label={e['label']} text='{e['text']}' method={e['method']}")
    return entities

#  Step 4: Extract key-value pairs from NER output
def extract_key_value_pairs(lines: List[List[Dict]], debug: bool = False) -> Dict[str, str]:
    """
    Build a simple key-value mapping from the NER entities.
    Strategy:
      - Run ner_classify to get spans.
      - For each label, choose the longest span (or aggregate for multiline fields).
      - If multiple candidate numeric / amount spans, pick one containing currency.
    """
    ents = ner_classify(lines, debug=debug)
    by_label: Dict[str, List[str]] = {}
    for e in ents:
        by_label.setdefault(e['label'], []).append(e['text'])

    results: Dict[str, str] = {}
    for label, texts in by_label.items():
        # Aggregate multiline fields
        if label in MULTILINE_FIELDS:
            # Deduplicate while preserving order
            seen = set()
            parts = []
            for t in texts:
                tt = t.strip()
                if tt.lower() in seen:
                    continue
                seen.add(tt.lower())
                parts.append(tt)
            results[label] = ' '.join(parts)
            continue
        # Amount preference: currency-containing token
        if label == 'amount':
            cur = [t for t in texts if re.search(r'(usd|hkd)', t, re.IGNORECASE)]
            cand = cur or texts
            results[label] = max(cand, key=len)
            continue
        # Date: pick earliest (assume first is correct) else longest
        if label == 'date':
            results[label] = texts[0]
            continue
        # Generic: longest span
        results[label] = max(texts, key=len)

    if debug:
        print(f"[DEBUG] NER-derived KV: {results}")
    return results

# Step 5: Postprocessing Cleaning
def clean_extracted_data(data: Dict[str, str]) -> Dict[str, str]:
    cleaned: Dict[str, str] = {}
    for k, v in data.items():
        val = re.sub(r'\s+', ' ', v).strip()

        # Mild field-specific tweaks (optional)
        if k == 'amount':
            # Try to pull first currency+number pattern without discarding original if fail
            m = re.search(r'(USD|HKD)[\s]*[\d,]+(?:\.\d{2})?', val, flags=re.IGNORECASE)
            if m:
                val = m.group(0).upper().replace('  ', ' ')

        if k in ('beneficiary_account', 'swift_code'):
            val = val.replace(' ', '').replace('|', '')

        cleaned[k] = val
    return cleaned

# CONTROL FLOW
def extract_table_from_pdf(file_path: str,
                           ocr_on_all: bool = False,
                           debug: bool = False) -> pd.DataFrame:
    boxes = extract_text_boxes(file_path, ocr_on_all=ocr_on_all, debug=debug)
    if not boxes:
        if debug:
            print("[DEBUG] No boxes extracted.")
        return pd.DataFrame()

    lines = cluster_into_lines(boxes)
    if debug:
        print(f"[DEBUG] Lines formed: {len(lines)}")

    raw = extract_key_value_pairs(lines, debug=debug)
    cleaned = clean_extracted_data(raw)

    if debug:
        print("[DEBUG] Raw:", raw)
        print("[DEBUG] Cleaned:", cleaned)

    return pd.DataFrame([cleaned]) if cleaned else pd.DataFrame()

# SAVE DATAFRAME TO CSV AND JSON
def save_dataframe_csv(df: pd.DataFrame, output_path: str, debug: bool = False) -> None:
    if df.empty:
        if debug:
            print(f"[DEBUG] Not saving CSV (empty): {output_path}")
        return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    if debug:
        print(f"[DEBUG] Saved CSV: {output_path}")

def save_dataframe_json(df: pd.DataFrame, output_path: str, debug: bool = False, indent: int = 2) -> None:
    if df.empty:
        if debug:
            print(f"[DEBUG] Not saving JSON (empty): {output_path}")
        return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient="records", indent=indent)
    if debug:
        print(f"[DEBUG] Saved JSON: {output_path}")

# INGEST DIRECTORY OF PDFs
def process_directory(pdf_dir: str,
                      output_dir: str = "extracted",
                      ocr_on_all: bool = False,
                      debug: bool = False) -> pd.DataFrame:
    """
    Process every PDF in a directory, saving per-file CSV/JSON and returning an aggregated DataFrame.
    Output naming: <stem>_extracted.(csv|json) inside output_dir.
    """
    pdf_paths = sorted(Path(pdf_dir).glob('*.pdf'))
    if not pdf_paths:
        if debug:
            print(f"[DEBUG] No PDFs found in {pdf_dir}")
        return pd.DataFrame()

    agg_rows = []
    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    for pdf in pdf_paths:
        if debug:
            print(f"[INFO] Processing PDF: {pdf.name}")
        df = extract_table_from_pdf(str(pdf), ocr_on_all=ocr_on_all, debug=debug)
        if df.empty:
            if debug:
                print(f"[WARN] No data extracted: {pdf.name}")
            continue
        # Add source file column for context
        df.insert(0, 'source_file', pdf.name)
        agg_rows.append(df)
        stem = pdf.stem
        csv_path = out_base / f"{stem}_extracted.csv"
        json_path = out_base / f"{stem}_extracted.json"
        save_dataframe_csv(df, str(csv_path), debug=debug)
        save_dataframe_json(df, str(json_path), debug=debug)

    if not agg_rows:
        return pd.DataFrame()
    agg = pd.concat(agg_rows, ignore_index=True)
    # Save aggregated outputs too
    save_dataframe_csv(agg, str(out_base / "all_extracted.csv"), debug=debug)
    save_dataframe_json(agg, str(out_base / "all_extracted.json"), debug=debug)
    return agg

# MAIN [only to test]
if __name__ == "__main__":
    sample_dir = "C:/Sakshi-Personal/Stripe_CodingTest/Sample Documents/"
    output_dir = "C:/Sakshi-Personal/Stripe_CodingTest/Output/"  # keep a central output
    agg_df = process_directory(sample_dir, output_dir=output_dir, debug=True)
    if agg_df.empty:
        print("No structured data extracted from any PDF.")
    else:
        print("\nAggregated DataFrame:")
        print(agg_df.to_string())
        print(f"\nPer-file and aggregated outputs stored in: {output_dir}")