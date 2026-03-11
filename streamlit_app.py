import json
from io import BytesIO

import streamlit as st

from core import process_pdf


st.set_page_config(page_title="PDF Date of Service Sorter", layout="wide")

st.title("PDF Date of Service Sorter")
st.write(
    "Upload a PDF, detect the Date of Service on each page, sort the pages, "
    "and download the reordered PDF, audit log, and review copy."
)

with st.expander("What this app does"):
    st.markdown(
        """
- Detects **Date of Service / Service Date / DOS**
- Works with **text PDFs**
- Uses **OCR fallback** for scanned pages
- Handles **multiple date candidates per page**
- Can choose the:
  - best contextual match
  - earliest date on page
  - latest date on page
- Produces:
  - sorted PDF
  - JSON audit log
  - review PDF with annotations
        """
    )

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

col1, col2, col3 = st.columns(3)

with col1:
    selection_mode = st.selectbox(
        "Date selection rule",
        options=["contextual", "earliest", "latest"],
        index=0,
        help="How the app chooses the page's DOS when multiple dates are found.",
    )

with col2:
    undated_position = st.selectbox(
        "Pages with no detected date",
        options=["last", "first"],
        index=0,
    )

with col3:
    use_ocr = st.checkbox(
        "Use OCR fallback for scanned pages",
        value=True,
        help="Recommended for scanned medical records.",
    )

strict_context = st.checkbox(
    "Prefer DOS context aggressively",
    value=True,
    help="Penalizes DOB, signature dates, printed dates, and similar non-service dates more strongly.",
)

if uploaded_file is not None:
    st.info(
        "Processing large scanned PDFs can take time, especially with OCR enabled."
    )

    if st.button("Process PDF", type="primary"):
        with st.spinner("Analyzing pages and sorting by Date of Service..."):
            pdf_bytes = uploaded_file.read()

            result = process_pdf(
                pdf_bytes=pdf_bytes,
                selection_mode=selection_mode,
                undated_position=undated_position,
                use_ocr=use_ocr,
                strict_context=strict_context,
            )

        st.success("Done.")

        summary = result["summary"]
        st.subheader("Summary")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Pages", summary["total_pages"])
        c2.metric("Dated Pages", summary["dated_pages"])
        c3.metric("Undated Pages", summary["undated_pages"])
        c4.metric("OCR Pages", summary["ocr_pages"])

        st.subheader("Preview")
        st.caption("First 20 pages from the audit log")

        preview_rows = []
        for item in result["audit"][:20]:
            preview_rows.append(
                {
                    "original_page": item["original_page"],
                    "chosen_date": item["chosen_date"],
                    "sort_key": item["sort_key"],
                    "used_ocr": item["used_ocr"],
                    "candidate_count": len(item["candidates"]),
                    "text_excerpt": item["text_excerpt"],
                }
            )

        st.dataframe(preview_rows, use_container_width=True)

        base_name = uploaded_file.name.rsplit(".", 1)[0]

        st.download_button(
            label="Download Sorted PDF",
            data=result["sorted_pdf_bytes"],
            file_name=f"{base_name}_sorted_by_dos.pdf",
            mime="application/pdf",
        )

        st.download_button(
            label="Download Audit JSON",
            data=result["audit_json_bytes"],
            file_name=f"{base_name}_dos_audit.json",
            mime="application/json",
        )

        st.download_button(
            label="Download Review PDF",
            data=result["review_pdf_bytes"],
            file_name=f"{base_name}_review_annotated.pdf",
            mime="application/pdf",
        )

        with st.expander("Full audit JSON"):
            st.code(
                json.dumps(result["audit"], indent=2),
                language="json",
            )
else:
    st.warning("Upload a PDF to begin.")
