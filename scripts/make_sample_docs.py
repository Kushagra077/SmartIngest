"""Generate realistic invoice sample documents (PNG + PDF) for multimodal tests.

These are *rendered* (not real third-party) documents, so there are no
copyright concerns, and the script is re-runnable. The PNG and PDF carry the
same content — a clean invoice from a whitelisted vendor that should
auto-approve — so they double as multimodal eval inputs for real Gemini.

Usage:
    uv run python scripts/make_sample_docs.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "samples"

# The invoice's ground-truth values (kept in sync with data/eval/multimodal.jsonl).
INVOICE = {
    "vendor": "ACME CORP",
    "address": "123 Industrial Way, Springfield, IL 62704",
    "gstin": "29ABCDE1234F1Z5",
    "number": "INV-2026-0042",
    "date": "2026-02-10",
    "due": "2026-03-12",
    "po": "PO-7781",
    "bill_to": "Globex Inc",
    "items": [
        ("Widget Pro (annual license)", "10", "50.00", "500.00"),
        ("Onboarding & setup", "1", "150.00", "150.00"),
    ],
    "subtotal": "650.00",
    "tax": "117.00",
    "total": "767.00",
    "terms": "Net 30",
}

_FONT_CANDIDATES = {
    "regular": [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    "bold": [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
}


def _font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES[kind]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def render() -> Image.Image:
    W, H = 850, 1100
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    ink, muted = (17, 17, 17), (90, 90, 90)
    margin = 60

    # Header
    d.text((margin, 50), INVOICE["vendor"], font=_font("bold", 40), fill=ink)
    d.text((margin, 100), INVOICE["address"], font=_font("regular", 18), fill=muted)
    d.text((margin, 126), f"GSTIN: {INVOICE['gstin']}", font=_font("regular", 18), fill=muted)
    d.text((W - margin - 200, 50), "INVOICE", font=_font("bold", 36), fill=ink)

    d.line((margin, 180, W - margin, 180), fill=(210, 210, 210), width=2)

    # Meta block
    meta = [
        ("Invoice #", INVOICE["number"]),
        ("Invoice Date", INVOICE["date"]),
        ("Due Date", INVOICE["due"]),
        ("PO Number", INVOICE["po"]),
        ("Bill To", INVOICE["bill_to"]),
    ]
    y = 210
    for label, value in meta:
        d.text((margin, y), f"{label}:", font=_font("bold", 18), fill=ink)
        d.text((margin + 160, y), value, font=_font("regular", 18), fill=ink)
        y += 32

    # Line-item table
    y += 24
    cols = [margin, 430, 560, 690]
    headers = ["Description", "Qty", "Unit Price", "Amount"]
    for x, h in zip(cols, headers):
        d.text((x, y), h, font=_font("bold", 18), fill=ink)
    y += 28
    d.line((margin, y, W - margin, y), fill=(210, 210, 210), width=1)
    y += 14
    for desc, qty, unit, amount in INVOICE["items"]:
        d.text((cols[0], y), desc, font=_font("regular", 18), fill=ink)
        d.text((cols[1], y), qty, font=_font("regular", 18), fill=ink)
        d.text((cols[2], y), unit, font=_font("regular", 18), fill=ink)
        d.text((cols[3], y), amount, font=_font("regular", 18), fill=ink)
        y += 32

    # Totals
    y += 30
    for label, value, bold in [
        ("Subtotal", INVOICE["subtotal"], False),
        ("Tax (18%)", INVOICE["tax"], False),
        ("Total", INVOICE["total"], True),
    ]:
        f = _font("bold", 20) if bold else _font("regular", 18)
        d.text((cols[2], y), f"{label}:", font=f, fill=ink)
        d.text((cols[3], y), value, font=f, fill=ink)
        y += 34

    d.text((margin, H - 90), f"Payment Terms: {INVOICE['terms']}",
           font=_font("regular", 18), fill=muted)
    d.text((margin, H - 60), "Thank you for your business.",
           font=_font("regular", 16), fill=muted)
    return img


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = render()
    png_path = OUT_DIR / "invoice_scan.png"
    pdf_path = OUT_DIR / "invoice_scan.pdf"
    img.save(png_path)
    img.save(pdf_path, "PDF", resolution=150.0)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
