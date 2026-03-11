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
    "mrn",
    "account number",
]

TOP_LEFT_WIDTH_RATIO = 0.45
TOP_LEFT_HEIGHT_RATIO = 0.35


@dataclass
class DateCandidate:
    raw_text: str
    normalized: str
    dt: datetime
    score: float
    start: int
    end: int
    context: str
    source_region: str  # "top_left", "full_page", "ocr_top_left", "ocr_full_page"


def normalize_year(dt: datetime) -> datetime:
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


def get_top_left_rect(page: fitz.Page) -> fitz.Rect:
    rect = page.rect
    return fitz.Rect(
        rect.x0,
        rect.y0,
        rect.x0 + rect.width * TOP_LEFT_WIDTH_RATIO,
        rect.y0 + rect.height * TOP_LEFT_HEIGHT_RATIO,
    )


def extract_page_text(page: fitz.Page) -> str:
    text = page.get_text("text")
    return collapse_whitespace(text)


def extract_top_left_text(page: fitz.Page) -> str:
    rect = get_top_left_rect(page)
    text = page.get_text("text", clip=rect)
    return collapse_whitespace(text)


def render_page_for_ocr(page: fitz.Page, scale: float = 2.0) -> Image.Image:
    matrix = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def crop_top_left_for_ocr(page: fitz.Page, scale: float = 2.0) -> Image.Image:
    rect = get_top_left_rect(page)
    matrix = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def ocr_image(img: Image.Image) -> str:
    text = pytesseract.image_to_string(img)
    return collapse_whitespace(text)


def ocr_page(page: fitz.Page) -> str:
    return ocr_image(render_page_for_ocr(page, scale=2.0))


def ocr_top_left(page: fitz.Page) -> str:
    return ocr_image(crop_top_left_for_ocr(page, scale=2.5))


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


def needs_ocr(text: str) -> bool:
    if not text:
        return True

    stripped = collapse_whitespace(text)
    if len(stripped) < 30:
        return True

    alnum_count = sum(ch.isalnum() for ch in stripped)
    return alnum_count < 15


def score_candidate(
    text: str,
    start: int,
    end: int,
    source_region: str,
    strict_context: bool = True,
) -> Tuple[float, str]:
    window_start = max(0, start - 120)
    window_end = min(len(text), end + 120)
    context = text[window_start:window_end].lower()
    before = text[max(0, start - 50):start].lower()

    score = 0.0

    for term in POSITIVE_TERMS:
        if term in context:
            score += 7.0

    for term in NEGATIVE_TERMS:
        if term in context:
            score -= 6.0

    if strict_context:
        for term in STRICT_NEGATIVE_TERMS:
            if term in context:
                score -= 4.0

    if any(term in before for term in POSITIVE_TERMS):
        score += 18.0

    if any(term in before for term in NEGATIVE_TERMS):
        score -= 18.0

    # Strong preference for top-left region
    if source_region in {"top_left", "ocr_top_left"}:
        score += 20.0
    else:
        score -= 4.0

    # Extra preference if the label is immediately before the date
    label_patterns = [
        r"(date of service|dos|service date)\s*[:\-]?\s*$",
    ]
    for pat in label_patterns:
        if re.search(pat, before, flags=re.IGNORECASE):
            score += 25.0

    # Penalize likely footer/header garbage
    footerish = ["printed on", "generated on", "page ", "electronically signed"]
    if any(term in context for term in footerish):
        score -= 10.0

    return score, collapse_whitespace(context)


def extract_date_candidates(
    text: str,
    source_region: str,
    strict_context: bool = True,
) -> List[DateCandidate]:
    candidates: List[DateCandidate] = []

    for raw, start, end in find_regex_date_matches(text):
        dt = safe_parse_date(raw)
        if not dt:
            continue

        if dt.year < 1900 or dt.year > datetime.now().year + 1:
            continue

        score, context = score_candidate(
            text=text,
            start=start,
            end=end,
            source_region=source_region,
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
                source_region=source_region,
            )
        )

    deduped = {}
    for c in candidates:
        key = (c.normalized, c.raw_text.lower(), c.source_region)
        if key not in deduped or c.score > deduped[key].score:
            deduped[key] = c

    return sorted(deduped.values(), key=lambda x: (x.score, x.dt), reverse=True)


def choose_candidate(candidates: List[DateCandidate], mode: str = "contextual") -> Optional[DateCandidate]:
    if not candidates:
        return None

    if mode == "earliest":
        return sorted(candidates, key=lambda x: x.dt)[0]

    if mode == "latest":
        return sorted(candidates, key=lambda x: x.dt)[-1]

    return sorted(candidates, key=lambda x: (x.score, x.dt), reverse=True)[0]


def gather_candidates_for_page(page: fitz.Page, use_ocr: bool, strict_context: bool) -> Tuple[List[DateCandidate], dict]:
    region_debug = {
        "top_left_text": "",
        "full_page_text": "",
        "ocr_top_left_text": "",
        "ocr_full_page_text": "",
        "used_ocr_top_left": False,
        "used_ocr_full_page": False,
    }

    all_candidates: List[DateCandidate] = []

    # 1. Top-left searchable text first
    top_left_text = extract_top_left_text(page)
    region_debug["top_left_text"] = top_left_text
    all_candidates.extend(
        extract_date_candidates(top_left_text, source_region="top_left", strict_context=strict_context)
    )

    # 2. If top-left is weak, OCR top-left
    if use_ocr and needs_ocr(top_left_text):
        ocr_tl = ocr_top_left(page)
        region_debug["ocr_top_left_text"] = ocr_tl
        if ocr_tl:
            region_debug["used_ocr_top_left"] = True
            all_candidates.extend(
                extract_date_candidates(ocr_tl, source_region="ocr_top_left", strict_context=strict_context)
            )

    # 3. Full page searchable text as fallback
    full_page_text = extract_page_text(page)
    region_debug["full_page_text"] = full_page_text
    all_candidates.extend(
        extract_date_candidates(full_page_text, source_region="full_page", strict_context=strict_context)
    )

    # 4. OCR full page only if needed
    if use_ocr and needs_ocr(full_page_text):
        ocr_fp = ocr_page(page)
        region_debug["ocr_full_page_text"] = ocr_fp
        if ocr_fp:
            region_debug["used_ocr_full_page"] = True
            all_candidates.extend(
                extract_date_candidates(ocr_fp, source_region="ocr_full_page", strict_context=strict_context)
            )

    # Deduplicate across regions, keeping best score
    best_by_date = {}
    for c in all_candidates:
        key = (c.normalized, c.raw_text.lower())
        if key not in best_by_date or c.score > best_by_date[key].score:
            best_by_date[key] = c

    final_candidates = sorted(best_by_date.values(), key=lambda x: (x.score, x.dt), reverse=True)
    return final_candidates, region_debug


def annotate_page(
    page: fitz.Page,
    chosen: Optional[DateCandidate],
    candidates: List[DateCandidate],
    searchable_text: str,
    used_ocr: bool,
) -> None:
    lines = []

    if chosen:
        lines.append(
            f"Chosen DOS: {chosen.normalized} ({chosen.raw_text}) | region={chosen.source_region} | score={chosen.score:.1f}"
        )
    else:
        lines.append("Chosen DOS: NONE")

    if candidates:
        short = ", ".join(
            [
                f"{c.normalized} [{c.raw_text}] region={c.source_region} score={c.score:.1f}"
                for c in candidates[:6]
            ]
        )
        lines.append(f"Candidates: {short}")
    else:
        lines.append("Candidates: NONE")

    lines.append(f"Used OCR: {used_ocr}")

    page.add_text_annot((36, 36), "\n".join(lines))

    if chosen and searchable_text:
        try:
            rects = page.search_for(chosen.raw_text, quads=False)
            for rect in rects[:10]:
                annot = page.add_highlight_annot(rect)
                annot.update()
        except Exception:
            pass


def build_sorted_pdf(source_doc: fitz.Document, sorted_indices: List[int]) -> bytes:
    out_doc = fitz.open()
    for index in sorted_indices:
        out_doc.insert_pdf(source_doc, from_page=index, to_page=index)
    data = out_doc.tobytes()
    out_doc.close()
    return data


def build_review_pdf(source_doc: fitz.Document, audit_rows: List[dict]) -> bytes:
    review_doc = fitz.open()
    review_doc.insert_pdf(source_doc)

    for row in audit_rows:
        page_idx = row["original_page"] - 1
        page = review_doc[page_idx]

        chosen = None
        if row["chosen_date"]:
            for c in row["candidates"]:
                if c["normalized"] == row["chosen_date"] and c["raw_text"] == row["chosen_raw_text"]:
                    chosen = DateCandidate(
                        raw_text=c["raw_text"],
                        normalized=c["normalized"],
                        dt=datetime.strptime(c["normalized"], "%Y-%m-%d"),
                        score=c["score"],
                        start=0,
                        end=0,
                        context=c["context"],
                        source_region=c["source_region"],
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
                source_region=c["source_region"],
            )
            for c in row["candidates"]
        ]

        annotate_page(
            page=page,
            chosen=chosen,
            candidates=candidates,
            searchable_text=row["searchable_text"],
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

        candidates, region_debug = gather_candidates_for_page(
            page=page,
            use_ocr=use_ocr,
            strict_context=strict_context,
        )

        chosen = choose_candidate(candidates, mode=selection_mode)

        used_ocr = region_debug["used_ocr_top_left"] or region_debug["used_ocr_full_page"]
        if used_ocr:
            ocr_pages += 1

        searchable_text = extract_page_text(page)

        audit_row = {
            "original_page": page_index + 1,
            "chosen_date": chosen.normalized if chosen else None,
            "chosen_raw_text": chosen.raw_text if chosen else None,
            "chosen_region": chosen.source_region if chosen else None,
            "sort_key": chosen.normalized if chosen else None,
            "used_ocr": used_ocr,
            "searchable_text": searchable_text,
            "text_excerpt": collapse_whitespace((region_debug["top_left_text"] or region_debug["ocr_top_left_text"] or searchable_text)[:250]),
            "candidates": [
                {
                    "raw_text": c.raw_text,
                    "normalized": c.normalized,
                    "score": round(c.score, 2),
                    "context": c.context,
                    "source_region": c.source_region,
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
