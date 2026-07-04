"""
ianvex - VEX V5 Engineering Notebook Evaluator
================================================
Upload a VEX Robotics V5 Engineering Notebook and get AI feedback grounded in
the official VEX Engineering Notebook Rubric and the current season game
"Override" game manual.

Core principle: the AI evaluates ONLY against the provided rubric and game
manual. It does not invent criteria, point values, or game rules.
"""

import base64
import html
import io
import os
import re
from pathlib import Path

import fitz  # PyMuPDF
import streamlit as st
from anthropic import Anthropic
from dotenv import load_dotenv
from PIL import Image
from pypdf import PdfReader, PdfWriter

load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
APP_DIR = Path(__file__).parent
REFERENCE_DIR = APP_DIR / "reference"

RUBRIC_PATH = REFERENCE_DIR / "rubric.pdf"
GAME_MANUAL_PATH = REFERENCE_DIR / "game_manual.pdf"

# Anthropic accepts at most 100 pages per PDF document block.
MAX_PDF_PAGES = 100

# Keep the whole request under Anthropic's ~32 MB limit. This budget is for the
# notebook images (base64), leaving headroom for the rubric PDF + manual text.
NOTEBOOK_B64_BUDGET = 24 * 1024 * 1024
# Long-edge pixel sizes to try, largest first; the app steps down until it fits.
RENDER_LADDER = [1700, 1500, 1300, 1100, 950, 800, 680]
JPEG_QUALITY = 80

MODELS = {
    "Claude Opus 4.8 (best quality, recommended)": "claude-opus-4-8",
    "Claude Sonnet 5 (faster, cheaper)": "claude-sonnet-5",
}
DEFAULT_MODEL_LABEL = "Claude Opus 4.8 (best quality, recommended)"

MAX_OUTPUT_TOKENS = 8000

SYSTEM_PROMPT = """\
You are an expert VEX Robotics V5 Engineering Notebook evaluator and mentor for \
the current competition season, whose game is "Override". You give students \
feedback on their Engineering Notebook entries.

You are given, as attached content:
  1. The official VEX Engineering Notebook Rubric (PDF).
  2. The "Override" game manual for this season (extracted text).
  3. The student's notebook entry/entries to evaluate.

STRICT GROUNDING RULES (never violate these):
- Base every judgment ONLY on the provided rubric and game manual. Do NOT invent \
rubric criteria, point values, categories, or game rules from memory.
- If the rubric or game manual is missing, incomplete, or does not cover \
something, say so plainly instead of guessing. Never fabricate a rule or a \
scoring criterion.
- When you make a point about the rubric, reference the specific rubric \
criterion / section it relates to (quote the rubric's wording where helpful).
- Do not claim the notebook contains something it does not, and do not assume \
content you cannot see.

HOW TO EVALUATE:
- Go through the rubric criterion by criterion. For each, state how the current \
entry measures up and what is missing.
- Clearly separate two kinds of advice:
    (A) WHAT TO CHANGE / IMPROVE in what is already written.
    (B) IDEAS FOR WHAT TO ADD that would strengthen the notebook.
- Be concrete and actionable. Prefer specific examples over vague praise.
- Assign every rubric score as a SINGLE whole number, never a range. If the \
rubric describes a level as a point band, pick the one number you judge most \
accurate.
- When you reference a page, use the page number written/printed on that notebook \
page if it is visible; if a page has no visible number, cite it by its position \
in the upload order and mark it "(as uploaded)".
- Where relevant, tie suggestions to Override game strategy, scoring, or \
constraints as described in the game manual.
- Use an encouraging, constructive mentor tone appropriate for students.

PAGE-FOCUS MODE:
- If the user's focus request names specific pages or a page range (e.g. "focus \
on pages 5-8"), LIMIT the detailed feedback to only those pages:
    * "Page-by-Page Comments", "Top Priorities: What to Change", "Ideas: What to \
Add", and "Override-Specific Suggestions" must address ONLY the requested pages.
    * "Quick Score Summary", "Overall Summary", and "Rubric-by-Rubric Evaluation" \
STILL cover the WHOLE notebook — rubric scoring is holistic and applies to the \
entire notebook, not a slice of it.
    * At the very top of the "Page-by-Page Comments" section, add a one-line note \
in italics: "Detailed feedback is limited to pages X-Y as requested; scores above \
reflect the whole notebook."
    * Match the requested page numbers to the numbers written on the notebook \
pages; if a page in the range has no visible number, go by upload order.
- If the user does NOT name specific pages, evaluate and comment on all pages \
normally.

OUTPUT FORMAT (Markdown):
## Quick Score Summary
A Markdown table with one row per rubric criterion, for an at-a-glance view.
Columns: | Rubric Criterion | Score | Why |
- Use the EXACT criterion names and the EXACT scoring scale / point values \
defined in the provided rubric (do not invent your own scale).
- Every score MUST be a single whole number, never a range (write 5, never "4-5").
- End the table with a bold **Total: X / Y** row, where Y is the maximum total \
points the rubric allows and X is the sum of your per-criterion scores.
- Add a bold one-line caveat under the table that these are estimates and only an \
official judge assigns real scores.

## Overall Summary
A short paragraph summarizing overall strengths and the biggest gaps. (Keep this \
general, whole-notebook view.)

## Rubric-by-Rubric Evaluation
For each rubric criterion: **[Criterion name] — [single number score]**, then \
bullets for what to improve.

## Page-by-Page Comments
Go through the notebook and, ONLY for pages that have something specific to change \
or add, give targeted feedback. Use this shape for each such page:
- **Page [N]:** [Change / Improve / Add] — [the specific action] *(Rubric: [the \
criterion it supports and why])*.
Cite the page number written on the notebook page when visible; otherwise use the \
upload position and mark it "(as uploaded)". Skip pages that need no change.

## Top Priorities: What to Change
Ranked fixes for the whole notebook (or only the focused pages if PAGE-FOCUS MODE \
applies). Give real detail for EACH item, 2-4 sentences: what is weak now, exactly \
what to change it to, where in the notebook it applies, and why it matters (cite \
the rubric criterion). Avoid one-line generic advice.

## Ideas: What to Add
Concrete additions for the whole notebook (or only the focused pages if PAGE-FOCUS \
MODE applies). For EACH idea, 2-4 sentences: what to add, what a strong version \
looks like (what it should contain), where it fits, and which rubric criterion it \
strengthens and how.

## Override-Specific Suggestions
Ideas tied directly to this season's game as described in the game manual.
"""


# --------------------------------------------------------------------------- #
# PDF / file helpers
# --------------------------------------------------------------------------- #
def guess_mime(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "")


@st.cache_data(show_spinner=False)
def extract_pdf_text(data: bytes) -> str:
    """Extract text from every page of a PDF, labeled by page number."""
    reader = PdfReader(io.BytesIO(data))
    parts = []
    for i, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        parts.append(f"[Page {i}]\n{text}")
    return "\n\n".join(parts)


def split_pdf(data: bytes, max_pages: int = MAX_PDF_PAGES):
    """Split a PDF's bytes into chunks of at most max_pages pages each."""
    reader = PdfReader(io.BytesIO(data))
    chunks = []
    for start in range(0, len(reader.pages), max_pages):
        writer = PdfWriter()
        for page in reader.pages[start:start + max_pages]:
            writer.add_page(page)
        buf = io.BytesIO()
        writer.write(buf)
        chunks.append(buf.getvalue())
    return chunks


def pdf_document_blocks(data: bytes, title: str, cache: bool = False):
    """One or more Anthropic 'document' blocks, splitting PDFs over the page cap."""
    reader = PdfReader(io.BytesIO(data))
    chunks = [data] if len(reader.pages) <= MAX_PDF_PAGES else split_pdf(data)

    blocks = []
    for idx, chunk in enumerate(chunks):
        b64 = base64.standard_b64encode(chunk).decode("utf-8")
        label = title if len(chunks) == 1 else f"{title} (part {idx + 1}/{len(chunks)})"
        blocks.append({
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": b64},
            "title": label,
        })
    if cache and blocks:
        blocks[-1]["cache_control"] = {"type": "ephemeral"}
    return blocks


def image_block(data: bytes, mime: str):
    b64 = base64.standard_b64encode(data).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}}


def _pdf_pages_to_jpegs(data, long_edge, quality):
    """Render each PDF page to a compressed JPEG (bytes)."""
    doc = fitz.open(stream=data, filetype="pdf")
    out = []
    try:
        for page in doc:
            rect = page.rect
            scale = min(long_edge / max(rect.width, rect.height), 4.0)
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            out.append(buf.getvalue())
    finally:
        doc.close()
    return out


def _image_to_jpeg(data, long_edge, quality):
    """Downscale an uploaded image and re-encode as JPEG (bytes)."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    scale = min(1.0, long_edge / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def _render_notebook(files_data, long_edge, quality):
    """files_data: tuple of (name, bytes). Returns a list of JPEG byte strings."""
    jpegs = []
    for name, data in files_data:
        mime = guess_mime(name)
        if mime == "application/pdf":
            jpegs.extend(_pdf_pages_to_jpegs(data, long_edge, quality))
        elif mime.startswith("image/"):
            jpegs.append(_image_to_jpeg(data, long_edge, quality))
    return jpegs


def build_notebook_blocks(uploaded_files):
    """Render the notebook to JPEG image blocks, shrinking until it fits the
    size budget. Returns (blocks, info)."""
    files_data = tuple((f.name, f.getvalue()) for f in uploaded_files)
    long_edge, jpegs, total_b64 = None, [], 0
    for long_edge in RENDER_LADDER:
        jpegs = _render_notebook(files_data, long_edge, JPEG_QUALITY)
        total_b64 = sum((len(b) + 2) // 3 * 4 for b in jpegs)  # base64 size
        if total_b64 <= NOTEBOOK_B64_BUDGET:
            break
    blocks = [image_block(b, "image/jpeg") for b in jpegs]
    info = {
        "pages": len(jpegs),
        "long_edge": long_edge,
        "mb": total_b64 / (1024 * 1024),
        "fit": total_b64 <= NOTEBOOK_B64_BUDGET,
    }
    return blocks, info


def load_reference_bytes(path: Path, uploaded):
    """Prefer an uploaded file; otherwise read from the reference/ folder."""
    if uploaded is not None:
        return uploaded.getvalue()
    if path.exists():
        return path.read_bytes()
    return None


# --------------------------------------------------------------------------- #
# Rendering: colored score boxes in the Quick Score Summary table
# --------------------------------------------------------------------------- #
def _score_style(n):
    """Box colors: low (0-1) red, mid (2-3) amber, high (4-5) green."""
    if n <= 1:
        return "background:#F8D7DA;color:#B02A37;border:1px solid #F1AEB5;"
    if n <= 3:
        return "background:#FFF3CD;color:#8A6100;border:1px solid #FFE69C;"
    return "background:#D1E7DD;color:#0F5132;border:1px solid #A3CFBB;"


def _chip(text, style):
    return ('<span style="display:inline-block;min-width:2.2em;text-align:center;'
            'padding:3px 12px;border-radius:8px;font-weight:700;'
            'font-variant-numeric:tabular-nums;' + style + '">' + text + '</span>')


def _md_inline(s):
    return re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html.escape(s))


def _split_row(row):
    return [c.strip() for c in row.strip().strip('|').split('|')]


def _is_separator(cells):
    return any('-' in c for c in cells) and all(re.fullmatch(r'[:\-\s]*', c) for c in cells)


def _score_table_html(table_lines):
    header = _split_row(table_lines[0])
    score_idx = next((k for k, h in enumerate(header) if 'score' in h.lower()), 1)
    rows = [c for c in (_split_row(ln) for ln in table_lines[1:]) if not _is_separator(c)]

    out = ['<table style="width:100%;border-collapse:collapse;">',
           '<thead><tr>']
    for h in header:
        out.append('<th style="text-align:left;padding:8px 10px;'
                   'border-bottom:2px solid rgba(128,128,128,.35);font-size:.78rem;'
                   'text-transform:uppercase;letter-spacing:.04em;opacity:.7;">'
                   + _md_inline(h) + '</th>')
    out.append('</tr></thead><tbody>')
    for cells in rows:
        is_total = any('total' in c.lower() for c in cells)
        out.append('<tr>')
        for k, c in enumerate(cells):
            style = 'padding:9px 10px;border-bottom:1px solid rgba(128,128,128,.18);vertical-align:middle;'
            if is_total:
                style += 'font-weight:700;border-top:2px solid rgba(128,128,128,.35);'
            if k == score_idx and not is_total and re.search(r'\d', c):
                n = int(re.search(r'-?\d+', c).group())
                out.append('<td style="' + style + '">' + _chip(str(n), _score_style(n)) + '</td>')
            else:
                out.append('<td style="' + style + '">' + _md_inline(c) + '</td>')
        out.append('</tr>')
    out.append('</tbody></table>')
    return '\n'.join(out)


def render_report(md):
    """Render the evaluation, drawing the Quick Score Summary scores as colored
    boxes. Falls back to plain markdown if anything unexpected happens."""
    try:
        lines = md.split('\n')
        start = next((i for i, ln in enumerate(lines)
                      if ln.strip().startswith('## ') and 'quick score summary' in ln.lower()), None)
        if start is None:
            st.markdown(md)
            return
        end = next((j for j in range(start + 1, len(lines))
                    if lines[j].strip().startswith('## ')), len(lines))

        before = '\n'.join(lines[:start]).strip()
        body = lines[start + 1:end]
        after = '\n'.join(lines[end:]).strip()
        table_lines = [ln for ln in body if ln.strip().startswith('|')]
        other = '\n'.join(ln for ln in body if not ln.strip().startswith('|')).strip()

        if before:
            st.markdown(before)
        st.markdown(lines[start])
        if len(table_lines) >= 2:
            st.markdown(_score_table_html(table_lines), unsafe_allow_html=True)
        else:
            st.markdown('\n'.join(body))
        if other:
            st.markdown(other)
        if after:
            st.markdown(after)
    except Exception:
        st.markdown(md)


def viewer_email():
    """The signed-in viewer's email on Streamlit Community Cloud (None locally)."""
    try:
        email = getattr(st.user, "email", None)
        if not email and hasattr(st.user, "get"):
            email = st.user.get("email")
        return email or None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="ianvex - Notebook Evaluator", page_icon="🤖", layout="wide")
st.title("🤖 ianvex — VEX V5 Engineering Notebook Evaluator")
st.caption("Feedback grounded in the official VEX Engineering Notebook Rubric and the *Override* game manual.")

with st.sidebar:
    _email = viewer_email()
    if _email:
        st.success(f"Signed in as {_email}")
    st.header("⚙️ Settings")

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            api_key = ""
    if api_key:
        st.success("API key loaded (server)")
    else:
        api_key = st.text_input("Anthropic API key", type="password",
                                help="Set ANTHROPIC_API_KEY in Streamlit secrets (deployed) "
                                     "or a .env file (local).")

    model_label = st.selectbox("Model", list(MODELS.keys()),
                               index=list(MODELS.keys()).index(DEFAULT_MODEL_LABEL))
    model_id = MODELS[model_label]

    st.divider()
    st.header("📚 Reference documents")
    st.caption("These ground every evaluation. Place them in the `reference/` "
               "folder as `rubric.pdf` and `game_manual.pdf`, or upload here.")

    rubric_upload = None
    if RUBRIC_PATH.exists():
        st.success("Rubric found: reference/rubric.pdf")
    else:
        st.warning("Rubric not found in reference/")
        rubric_upload = st.file_uploader("Upload rubric (PDF)", type=["pdf"], key="rubric")

    manual_upload = None
    if GAME_MANUAL_PATH.exists():
        st.success("Game manual found: reference/game_manual.pdf")
    else:
        st.warning("Game manual not found in reference/")
        manual_upload = st.file_uploader("Upload Override game manual (PDF)", type=["pdf"], key="manual")

st.subheader("1. Upload the notebook to evaluate")
notebook_files = st.file_uploader(
    "Upload your notebook as a PDF and/or page images (PNG/JPG/WEBP). "
    "You can drag in multiple files at once.",
    type=["pdf", "png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)

st.subheader("2. Optional focus")
focus = st.text_area(
    "Anything specific you want the agent to focus on? (optional)",
    placeholder="e.g. Focus on pages 5-8   —or—   Focus on the testing documentation.",
    height=80,
)
st.caption("Tip: name pages (e.g. \"focus on pages 5-8\") to limit the detailed "
           "feedback to just those pages. The score and total still cover the "
           "whole notebook.")

run = st.button("🔍 Evaluate notebook", type="primary", use_container_width=True)


# --------------------------------------------------------------------------- #
# Run evaluation
# --------------------------------------------------------------------------- #
def build_content(rubric_bytes, manual_bytes, nb_blocks, focus_text):
    content = []

    # 1. Rubric as PDF (small — keeps the scoring-table layout intact).
    content.extend(pdf_document_blocks(rubric_bytes, "VEX Engineering Notebook Rubric"))

    # 2. Game manual as extracted text (too long to attach as a PDF).
    #    cache_control here marks a cache breakpoint covering rubric + manual.
    manual_text = extract_pdf_text(manual_bytes)
    content.append({
        "type": "text",
        "text": "--- OVERRIDE GAME MANUAL (extracted text) ---\n\n" + manual_text,
        "cache_control": {"type": "ephemeral"},
    })

    # 3. The student's notebook (pre-rendered image blocks).
    content.append({"type": "text",
                    "text": "--- Below is the student's Engineering Notebook to evaluate ---"})
    content.extend(nb_blocks)

    # 4. The instruction.
    instruction = ("Evaluate the student's Engineering Notebook above against the "
                   "VEX Engineering Notebook Rubric, specific to the Override game. "
                   "Follow the output format exactly.")
    if focus_text.strip():
        instruction += ("\n\nUSER FOCUS REQUEST: " + focus_text.strip() +
                        "\nIf this request names specific pages or a page range, "
                        "apply PAGE-FOCUS MODE exactly as described in the system "
                        "prompt (limit detailed feedback to those pages; keep "
                        "scoring for the whole notebook).")
    content.append({"type": "text", "text": instruction})
    return content


if run:
    if not api_key:
        st.error("Please provide an Anthropic API key (sidebar or .env).")
        st.stop()

    rubric_bytes = load_reference_bytes(RUBRIC_PATH, rubric_upload)
    if rubric_bytes is None:
        st.error("No rubric available. Add reference/rubric.pdf or upload it in the sidebar.")
        st.stop()

    manual_bytes = load_reference_bytes(GAME_MANUAL_PATH, manual_upload)
    if manual_bytes is None:
        st.error("No game manual available. Add reference/game_manual.pdf or upload it in the sidebar.")
        st.stop()

    if not notebook_files:
        st.error("Please upload at least one notebook file to evaluate.")
        st.stop()

    try:
        client = Anthropic(api_key=api_key)

        with st.spinner("Preparing notebook pages..."):
            nb_blocks, info = build_notebook_blocks(notebook_files)
        st.caption(f"Prepared {info['pages']} notebook page(s) at ~{info['long_edge']}px "
                   f"· {info['mb']:.1f} MB request payload.")
        if not info["fit"]:
            st.warning("Your notebook is very large. Pages were compressed to the "
                       "smallest setting but may still exceed the size limit. If you "
                       "hit an error, upload fewer pages at a time.")

        content = build_content(rubric_bytes, manual_bytes, nb_blocks, focus)

        with st.spinner("Evaluating against the rubric and Override game manual..."):
            resp = client.messages.create(
                model=model_id,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
        result = "".join(block.text for block in resp.content if block.type == "text")

        st.divider()
        st.subheader("📋 Evaluation")
        render_report(result)

        st.download_button("💾 Download feedback (Markdown)", result,
                           file_name="notebook_feedback.md", mime="text/markdown")
    except Exception as e:
        st.error(f"Something went wrong: {e}")


# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #
st.markdown(
    "<div style='text-align:center;color:rgba(128,128,128,.7);font-size:.72rem;"
    "margin-top:2.5rem;padding-top:0.9rem;border-top:1px solid rgba(128,128,128,.2);'>"
    "© 2026 Ian Shin · Published July 4, 2026</div>",
    unsafe_allow_html=True,
)
