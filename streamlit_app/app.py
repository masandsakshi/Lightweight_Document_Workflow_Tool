import streamlit as st
import json
import pandas as pd
from helpers import upload_pdf, extract, download, pretty_json

st.set_page_config(page_title="PDF Extraction", layout="wide")

st.title("PDF Extraction Demo")
st.markdown("Upload a PDF, extract structured fields, correct them, and download.")

with st.sidebar:
	st.header("1. Upload PDF")
	uploaded = st.file_uploader("Choose PDF", type=["pdf"])   
	run_extract = st.button("Run Extraction", type="primary", disabled=uploaded is None)

if "session" not in st.session_state:
	st.session_state.session = {}

col1, col2 = st.columns(2, gap="large")

if run_extract and uploaded:
	with st.spinner("Uploading..."):
		meta = upload_pdf(uploaded.getvalue(), uploaded.name)
	st.session_state.session['file_id'] = meta['file_id']
	st.session_state.session['filename'] = uploaded.name
	# Create a stable per-file key (changes when a different file is processed)
	new_file_key = f"{meta['file_id']}::{uploaded.name}"
	old_file_key = st.session_state.session.get('active_file_key')
	with st.spinner("Extracting..."):
		# Force OCR option removed; default behavior (machine text first, fallback to OCR)
		result = extract(meta['file_id'], uploaded.name, ocr_on_all=False)
	st.session_state.session['raw_result'] = result
	st.session_state.session['active_file_key'] = new_file_key
	# If this is a different file than last time, clear prior edit state
	if old_file_key and old_file_key != new_file_key:
		for k in ['corrected', 'csv_edit_df']:
			st.session_state.session.pop(k, None)
	st.success("Extraction complete")
	st.success("Extraction complete")

if 'raw_result' in st.session_state.session:
	result = st.session_state.session['raw_result']
	data = result.get('data', [])
	original_json = pretty_json(data)

	with col1:
		st.subheader("Extracted Data (Original)")
		st.code(original_json, language="json")
		if data:
			df_view = pd.DataFrame(data)
			st.dataframe(df_view, use_container_width=True)
			col_dl1, col_dl2 = st.columns(2)
			with col_dl1:
				st.download_button(
					"Download Original JSON",
					data=original_json,
					file_name="extracted_original.json",
					mime="application/json",
					key="dl_original_json"
				)
			with col_dl2:
				st.download_button(
					"Download Original CSV",
					data=df_view.to_csv(index=False),
					file_name="extracted_original.csv",
					mime="text/csv",
					key="dl_original_csv"
				)

	with col2:
		st.subheader("Edit & Correct Data")
		if not data:
			st.info("Upload and extract a PDF to begin editing.")
		else:
			file_key = st.session_state.session.get('active_file_key', 'current')
			tabs = st.tabs(["JSON Editor", "CSV Editor", "Preview / Download"])

			with tabs[0]:
				edited = st.text_area(
					"Modify JSON then click Apply",
					value=original_json,
					height=360,
					key=f"json_editor_area_{file_key}"
				)
				if st.button("Apply JSON Changes", key=f"apply_json_{file_key}"):
					try:
						parsed = json.loads(edited)
						if not isinstance(parsed, list):
							st.warning("Expected a list of records (list[dict]).")
						else:
							st.session_state.session['corrected'] = parsed
							st.success("JSON corrections applied.")
					except json.JSONDecodeError as e:
						st.error(f"Invalid JSON: {e}")

			with tabs[1]:
				csv_state_key = f"csv_edit_df_{file_key}"
				if csv_state_key not in st.session_state.session:
					st.session_state.session[csv_state_key] = pd.DataFrame(data)
				csv_df = st.data_editor(
					st.session_state.session[csv_state_key],
					use_container_width=True,
					num_rows="dynamic",
					key=f"csv_editor_{file_key}"
				)
				if st.button("Apply CSV Changes", key=f"apply_csv_{file_key}"):
					st.session_state.session['corrected'] = csv_df.to_dict(orient='records')
					st.session_state.session[csv_state_key] = csv_df
					st.success("CSV corrections applied.")

			with tabs[2]:
				if 'corrected' not in st.session_state.session:
					st.info("Apply JSON or CSV changes to see a corrected preview.")
				else:
					corrected_list = st.session_state.session['corrected']
					if isinstance(corrected_list, list) and corrected_list:
						df_corr = pd.DataFrame(corrected_list)
						st.dataframe(df_corr, use_container_width=True)
						corr_json = pretty_json(corrected_list)
						c1, c2 = st.columns(2)
						with c1:
							st.download_button(
								"Download Corrected JSON",
								data=corr_json,
								file_name="extracted_corrected.json",
								mime="application/json",
								key="dl_corrected_json"
							)
						with c2:
							st.download_button(
								"Download Corrected CSV",
								data=df_corr.to_csv(index=False),
								file_name="extracted_corrected.csv",
								mime="text/csv",
								key="dl_corrected_csv"
							)

st.markdown("---")
st.caption("Backend: FastAPI | Frontend: Streamlit | Extraction heuristic pipeline (regex + fuzzy + context)")
