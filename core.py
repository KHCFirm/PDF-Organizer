import json
import re
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
import pytesseract
from dateutil import parser
from PIL import Image


DATE_PATTERNS = [
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
    r"\b(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\s+\d{1,2},\s+\d{2,4}\b",
    r"\b\d{1,2}\s+(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\s+\d{2,4}\b",
]

POSITIVE_TERMS = [
    "date of service",
    "dos",
    "service date",
    "date(s) of service",
    "treatment date",
    "visit date",
    "encounter date",
    "seen on",
]

NEGATIVE_TERMS = [
    "dob",
    "date of birth",
    "birth date",
    "signed",
    "signature",
    "electronically signed",
    "printed",
    "print date",
    "printed on",
    "generated",
    "report date",
    "statement date",
    "invoice date",
    "payment date",
    "posted",
    "created",
    "modified",
    "admission date",
    "discharge date",
]

STRICT_NEGATIVE_TERMS = [
    "dob:",
    "date of birth:",
    "signed by",
    "signature date",
    "printed:",
    "printed on:",
    "generated on",
    "page ",
]

MONTH_WORDS = [
    "jan", "january", "feb", "february", "mar", "march", "apr", "april", "may",
    "jun", "june", "jul", "july", "aug", "august", "sep", "sept", "september",
    "oct", "october", "nov", "november", "dec", "december",
]


@dataclass
class DateCandidate:
    raw_text: str
    normalized: str
    dt: datetime
    score: float
    start: int
    end: int
    context: str


def normalize_year(dt: datetime) -> datetime:
    # Helps with ambiguous 2-digit years parsed too far in the future.
    if dt.year > datetime.now().year + 1:
        return dt.replace(year=dt.year - 100)
    return dt


def safe_parse_date(text: str) -> Optional[datetime]:
    try:
        dt = parser.parse(text, fuzzy=False, dayfirst=False, yearfirst=False)
        dt = normalize_year(dt)
        return dt
    except Exception:
        return None


def normalize_date_string(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_page_text(page: fitz.Page) -> str:
    text = page.get_text("text")
    return collapse_whitespace(text)


def render_page_for_ocr(page: fitz.Page, scale: float = 2.0) -> Image.Image:
    matrix = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    mode = "RGB"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    return img


def ocr_page(page: fitz.Page) -> str:
    img = render_page_for_ocr(page, scale=2.0)
    text = pytesseract.image_to_string(img)
    return collapse_whitespace(text)


def find_regex_date_matches(text: str) -> List[Tuple[str, int, int]]:
    matches = []
    lowered = text.lower()

    for pattern in DATE_PATTERNS:
        for m in re.finditer(pattern, lowered, flags=re.IGNORECASE):
            matches.append((text[m.start():m.end()], m.start(), m.end()))

    seen = set()
    deduped = []
    for raw, start, end in sorted(matches, key=lambda x: (x[1], x[2])):
        key = (raw.lower(), start, end)
        if key not in seen:
            seen.add(key)
            deduped.append((raw, start, end))
    return deduped


def score_candidate(
    text: str,
    start: int,
    end: int,
    strict_context: bool = True,
) -> Tuple[float, str]:
    window_start = max(0, start - 120)
    window_end = min(len(text), end + 120)
    context = text[window_start:window_end].lower()

    score = 0.0

    for term in POSITIVE_TERMS:
        if term in context:
            score += 5.0

    for term in NEGATIVE_TERMS:
        if term in context:
            score -= 4.0

    if strict_context:
        for term in STRICT_NEGATIVE_TERMS:
            if term in context:
                score -= 3.0

    before = text[max(0, start - 40):start].lower()
    if any(term in before for term in POSITIVE_TERMS):
        score += 8.0

    if any(term in before for term in NEGATIVE_TERMS):
        score -= 8.0

    return score, collapse_whitespace(context)


def extract_date_candidates(text: str, strict_context: bool = True) -> List[DateCandidate]:
    candidates: List[DateCandidate] = []

    for raw, start, end in find_regex_date_matches(text):
        dt = safe_parse_date(raw)
        if not dt:
            continue

        # Filter obviously unreasonable dates for medical record sorting.
        if dt.year < 1900 or dt.year > datetime.now().year + 1:
            continue

        score, context = score_candidate(
            text=text,
            start=start,
            end=end,
            strict_context=strict_context,
        )

        candidates.append(
            DateCandidate(
                raw_text=raw,
                normalized=normalize_date_string(dt),
                dt=dt,
                score=score,
                start=start,
                end=end,
                context=context,
            )
        )

    deduped = {}
    for c in candidates:
        key = (c.normalized, c.raw_text.lower())
        if key not in deduped or c.score > deduped[key].score:
            deduped[key] = c

    return sorted(
        deduped.values(),
        key=lambda x: (x.score, x.dt),
        reverse=True,
    )


def choose_candidate(
    candidates: List[DateCandidate],
    mode: str = "contextual",
) -> Optional[DateCandidate]:
    if not candidates:
        return None

    if mode == "earliest":
        return sorted(candidates, key=lambda x: x.dt)[0]

    if mode == "latest":
        return sorted(candidates, key=lambda x: x.dt)[-1]

    # contextual
    return sorted(candidates, key=lambda x: (x.score, x.dt), reverse=True)[0]


def needs_ocr(text: str) -> bool:
    if not text:
        return True

    stripped = collapse_whitespace(text)
    if len(stripped) < 40:
        return True

    alnum_count = sum(ch.isalnum() for ch in stripped)
    return alnum_count < 20


def annotate_page(
    page: fitz.Page,
    chosen: Optional[DateCandidate],
    candidates: List[DateCandidate],
    original_text: str,
    used_ocr: bool,
) -> None:
    note_lines = []

    if chosen:
        note_lines.append(f"Chosen DOS: {chosen.normalized} ({chosen.raw_text})")
    else:
        note_lines.append("Chosen DOS: NONE")

    if candidates:
        candidate_text = ", ".join(
            [f"{c.normalized} [{c.raw_text}] score={c.score:.1f}" for c in candidates[:8]]
        )
        note_lines.append(f"Candidates: {candidate_text}")
    else:
        note_lines.append("Candidates: NONE")

    note_lines.append(f"Used OCR: {used_ocr}")

    note = "\n".join(note_lines)

    # Put a small note near top-left.
    page.add_text_annot((36, 36), note)

    if chosen and original_text:
        # Highlight exact chosen raw text matches if searchable text exists.
        try:
            rects = page.search_for(chosen.raw_text, quads=False)
            for rect in rects[:10]:
                annot = page.add_highlight_annot(rect)
                annot.update()
        except Exception:
            pass


def build_sorted_pdf(
    source_doc: fitz.Document,
    sorted_indices: List[int],
) -> bytes:
    out_doc = fitz.open()
    for index in sorted_indices:
        out_doc.insert_pdf(source_doc, from_page=index, to_page=index)
    data = out_doc.tobytes()
    out_doc.close()
    return data


def build_review_pdf(
    source_doc: fitz.Document,
    audit_rows: List[dict],
) -> bytes:
    review_doc = fitz.open()
    review_doc.insert_pdf(source_doc)

    for row in audit_rows:
        page_idx = row["original_page"] - 1
        page = review_doc[page_idx]

        chosen = None
        if row["chosen_date"]:
            for c in row["candidates"]:
                if c["normalized"] == row["chosen_date"]:
                    chosen = DateCandidate(
                        raw_text=c["raw_text"],
                        normalized=c["normalized"],
                        dt=datetime.strptime(c["normalized"], "%Y-%m-%d"),
                        score=c["score"],
                        start=0,
                        end=0,
                        context=c["context"],
                    )
                    break

        candidates = [
            DateCandidate(
                raw_text=c["raw_text"],
                normalized=c["normalized"],
                dt=datetime.strptime(c["normalized"], "%Y-%m-%d"),
                score=c["score"],
                start=0,
                end=0,
                context=c["context"],
            )
            for c in row["candidates"]
        ]

        annotate_page(
            page=page,
            chosen=chosen,
            candidates=candidates,
            original_text=row["searchable_text"],
            used_ocr=row["used_ocr"],
        )

    data = review_doc.tobytes()
    review_doc.close()
    return data


def process_pdf(
    pdf_bytes: bytes,
    selection_mode: str = "contextual",
    undated_position: str = "last",
    use_ocr: bool = True,
    strict_context: bool = True,
) -> dict:
    source_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    audit_rows = []
    sortable_rows = []
    ocr_pages = 0

    for page_index in range(len(source_doc)):
        page = source_doc[page_index]
        text = extract_page_text(page)
        searchable_text = text
        used_ocr = False

        if use_ocr and needs_ocr(text):
            ocr_text = ocr_page(page)
            if ocr_text:
                text = ocr_text
                used_ocr = True
                ocr_pages += 1

        candidates = extract_date_candidates(text, strict_context=strict_context)
        chosen = choose_candidate(candidates, mode=selection_mode)

        audit_row = {
            "original_page": page_index + 1,
            "chosen_date": chosen.normalized if chosen else None,
            "sort_key": chosen.normalized if chosen else None,
            "used_ocr": used_ocr,
            "searchable_text": searchable_text,
            "text_excerpt": collapse_whitespace(text[:250]),
            "candidates": [
                {
                    "raw_text": c.raw_text,
                    "normalized": c.normalized,
                    "score": round(c.score, 2),
                    "context": c.context,
                }
                for c in candidates
            ],
        }
        audit_rows.append(audit_row)

        sortable_rows.append(
            {
                "page_index": page_index,
                "date_obj": chosen.dt if chosen else None,
                "date_str": chosen.normalized if chosen else None,
                "original_page": page_index + 1,
            }
        )

    if undated_position == "first":
        sorted_rows = sorted(
            sortable_rows,
            key=lambda r: (
                r["date_obj"] is not None,
                r["date_obj"] or datetime.min,
                r["original_page"],
            ),
        )
    else:
        sorted_rows = sorted(
            sortable_rows,
            key=lambda r: (
                r["date_obj"] is None,
                r["date_obj"] or datetime.max,
                r["original_page"],
            ),
        )

    sorted_indices = [r["page_index"] for r in sorted_rows]

    sorted_pdf_bytes = build_sorted_pdf(source_doc, sorted_indices)
    review_pdf_bytes = build_review_pdf(source_doc, audit_rows)
    audit_json_bytes = json.dumps(audit_rows, indent=2).encode("utf-8")

    dated_pages = sum(1 for row in audit_rows if row["chosen_date"])
    undated_pages = len(audit_rows) - dated_pages

    source_doc.close()

    return {
        "sorted_pdf_bytes": sorted_pdf_bytes,
        "review_pdf_bytes": review_pdf_bytes,
        "audit_json_bytes": audit_json_bytes,
        "audit": audit_rows,
        "summary": {
            "total_pages": len(audit_rows),
            "dated_pages": dated_pages,
            "undated_pages": undated_pages,
            "ocr_pages": ocr_pages,
        },
    }
