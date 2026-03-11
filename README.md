PDF Date of Service Sorter

A Streamlit app that opens a PDF, detects the Date of Service on each page, sorts the pages, and lets you download:

- a reordered PDF
- a JSON audit log
- a review PDF with annotations

Features

- Works with text-searchable PDFs
- Uses OCR fallback for scanned PDFs
- Detects multiple date candidates per page
- Can choose:
  - contextual best match
  - earliest date
  - latest date
- Adds review annotations for QA

Files

- `streamlit_app.py` - Streamlit UI
- `core.py` - PDF processing and DOS detection logic
- `requirements.txt` - Python dependencies
- `packages.txt` - system package for Streamlit Cloud OCR

Local Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
