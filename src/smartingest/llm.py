"""LLM client abstraction for the multimodal vision + extraction calls.

Two implementations share a common :class:`LLMClient` protocol:

* :class:`GeminiClient` — calls the Gemini API for real multimodal
  classification and extraction.
* :class:`MockLLMClient` — a deterministic, offline stub that inspects text
  documents heuristically. It lets the whole pipeline (and the test-suite, and
  demos) run with no API key or network access.

:func:`get_llm_client` picks the right one based on settings.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol

from smartingest.config import Settings
from smartingest.logging_config import get_logger
from smartingest.models import (
    ClassificationResult,
    DocumentType,
    ExtractedFields,
    LineItem,
)

logger = get_logger(__name__)


class LLMError(RuntimeError):
    """Raised when an LLM call fails irrecoverably."""


class LLMClient(Protocol):
    """Interface implemented by all LLM backends."""

    def classify(self, file_path: str, mime_type: str) -> ClassificationResult:
        """Classify a document into a :class:`DocumentType`."""

    def extract(
        self, file_path: str, mime_type: str, document_type: DocumentType
    ) -> tuple[ExtractedFields, float]:
        """Extract structured fields. Returns (fields, confidence)."""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = """You are a document classifier. Examine the document and \
respond with ONLY a JSON object of the form:
{{"document_type": "<invoice|contract|resume|id_document|unknown>", \
"confidence": <float 0-1>, "reasoning": "<short reason>"}}
"""

_EXTRACT_PROMPT = """You are a precise document data extractor. The document is \
a {doc_type}. Extract the relevant fields and respond with ONLY a JSON object \
matching this schema (omit fields that are not present):
{schema}
Also include a top-level "confidence" float (0-1) reflecting extraction quality.
"""

# Per-document-type schema hints. Sending only the relevant subset to the model
# keeps the prompt focused and avoids cross-type field bleed.
_INVOICE_SCHEMA = {
    "vendor_name": "string",
    "invoice_number": "string",
    "invoice_date": "YYYY-MM-DD",
    "due_date": "YYYY-MM-DD",
    "po_number": "string",
    "currency": "string (ISO code, e.g. INR/USD)",
    "vendor_address": "string",
    "vendor_tax_id": "string (GSTIN/VAT/EIN)",
    "vendor_bank_details": "string (account/IBAN/IFSC)",
    "bill_to": "string",
    "ship_to": "string",
    "line_items": [
        {
            "description": "string",
            "quantity": 0,
            "unit_price": 0,
            "line_total": 0,
            "hsn_sac_code": "string",
            "tax_rate": 0,
        }
    ],
    "subtotal": "number",
    "tax_amount": "number",
    "cgst": "number (Indian GST only)",
    "sgst": "number (Indian GST only)",
    "igst": "number (Indian GST only)",
    "discount": "number",
    "shipping": "number",
    "grand_total": "number",
    "payment_terms": "string",
    "payment_method": "string",
}

_CONTRACT_SCHEMA = {
    "party_names": ["string"],
    "party_roles": {"<party name>": "client|vendor|other"},
    "effective_date": "YYYY-MM-DD",
    "expiration_date": "YYYY-MM-DD",
    "term_length": "string",
    "renewal_type": "auto|manual",
    "notice_period": "string",
    "contract_value": "number",
    "currency": "string",
    "payment_schedule": "string",
    "termination_clause": "short snippet or flag",
    "liability_cap": "short snippet or flag",
    "confidentiality": "short snippet or flag",
    "governing_law": "string",
    "jurisdiction": "string",
    "signatures_present": "boolean",
    "signatory_names": ["string"],
}

_RESUME_SCHEMA = {
    "full_name": "string",
    "candidate_name": "string",
    "email": "string",
    "phone": "string",
    "location": "string",
    "links": ["string (LinkedIn/GitHub/portfolio URLs)"],
    "total_years_experience": "number (derived)",
    "work_history": [
        {
            "company": "string",
            "title": "string",
            "start_date": "YYYY-MM-DD or YYYY-MM",
            "end_date": "YYYY-MM-DD or 'present'",
            "responsibilities": ["string"],
        }
    ],
    "education": [
        {
            "institution": "string",
            "degree": "string",
            "field": "string",
            "graduation_year": "YYYY",
        }
    ],
    "skills": ["string"],
    "certifications": ["string"],
    "languages": ["string"],
}

_ID_SCHEMA = {
    "id_type": "passport|drivers_license|aadhaar|pan|other",
    "full_name": "string",
    "id_number": "string",
    "date_of_birth": "YYYY-MM-DD",
    "expiry_date": "YYYY-MM-DD",
    "issuing_authority": "string",
    "nationality": "string",
}

_SCHEMA_BY_TYPE: dict[DocumentType, dict] = {
    DocumentType.INVOICE: _INVOICE_SCHEMA,
    DocumentType.CONTRACT: _CONTRACT_SCHEMA,
    DocumentType.RESUME: _RESUME_SCHEMA,
    DocumentType.ID_DOCUMENT: _ID_SCHEMA,
}


def _schema_for(document_type: DocumentType) -> dict:
    """Return the field-schema hint for a document type (invoice as fallback)."""
    return _SCHEMA_BY_TYPE.get(document_type, _INVOICE_SCHEMA)


# ---------------------------------------------------------------------------
# Gemini implementation
# ---------------------------------------------------------------------------


class GeminiClient:
    """Real Gemini-backed client using the ``google-genai`` SDK."""

    def __init__(self, api_key: str, model: str) -> None:
        try:
            from google import genai  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise LLMError(
                "google-genai is not installed; run `pip install google-genai`"
            ) from exc

        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def _read_part(self, file_path: str, mime_type: str):
        """Build a Gemini content part from a file on disk."""
        from google.genai import types  # type: ignore

        try:
            data = Path(file_path).read_bytes()
        except OSError as exc:
            raise LLMError(f"Failed to read file {file_path}: {exc}") from exc
        return types.Part.from_bytes(data=data, mime_type=mime_type)

    def _generate_json(self, prompt: str, file_path: str, mime_type: str) -> dict:
        """Call Gemini requesting a JSON response and parse it."""
        from google.genai import types  # type: ignore

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=[prompt, self._read_part(file_path, mime_type)],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
        except Exception as exc:  # pragma: no cover - network failure path
            raise LLMError(f"Gemini request failed: {exc}") from exc

        return _safe_json_loads(response.text or "{}")

    def classify(self, file_path: str, mime_type: str) -> ClassificationResult:
        payload = self._generate_json(_CLASSIFY_PROMPT, file_path, mime_type)
        return _classification_from_payload(payload)

    def extract(
        self, file_path: str, mime_type: str, document_type: DocumentType
    ) -> tuple[ExtractedFields, float]:
        prompt = _EXTRACT_PROMPT.format(
            doc_type=document_type.value,
            schema=json.dumps(_schema_for(document_type), indent=2),
        )
        payload = self._generate_json(prompt, file_path, mime_type)
        return _fields_from_payload(payload)


# ---------------------------------------------------------------------------
# Mock implementation
# ---------------------------------------------------------------------------


class MockLLMClient:
    """Deterministic offline client.

    Reads the document as UTF-8 text (binary content degrades gracefully) and
    applies keyword/regex heuristics. This is *not* meant to be accurate — it
    exists so the pipeline is runnable and testable without Gemini.
    """

    def _read_text(self, file_path: str) -> str:
        try:
            return Path(file_path).read_text(encoding="utf-8", errors="ignore")
        except OSError:  # pragma: no cover - defensive
            return ""

    def classify(self, file_path: str, mime_type: str) -> ClassificationResult:
        text = self._read_text(file_path).lower()
        scores = {
            DocumentType.INVOICE: _count(text, ["invoice", "bill to", "amount due", "total"]),
            DocumentType.CONTRACT: _count(text, ["agreement", "party", "hereby", "terms"]),
            DocumentType.RESUME: _count(text, ["resume", "experience", "education", "skills"]),
            DocumentType.ID_DOCUMENT: _count(text, ["passport", "id number", "date of birth"]),
        }
        best = max(scores, key=scores.get)
        hits = scores[best]
        if hits == 0:
            return ClassificationResult(
                document_type=DocumentType.UNKNOWN,
                confidence=0.3,
                reasoning="No recognisable keywords found.",
            )
        confidence = min(0.5 + 0.15 * hits, 0.99)
        return ClassificationResult(
            document_type=best,
            confidence=round(confidence, 2),
            reasoning=f"Matched {hits} keyword(s) for {best.value}.",
        )

    def extract(
        self, file_path: str, mime_type: str, document_type: DocumentType
    ) -> tuple[ExtractedFields, float]:
        text = self._read_text(file_path)
        fields = ExtractedFields()
        confidence = 0.6

        if document_type == DocumentType.INVOICE:
            fields.vendor_name = _first_match(text, r"(?:vendor|from)\s*[:\-]\s*(.+)")
            fields.invoice_number = _first_match(text, r"invoice\s*#\s*:?\s*(\S+)")
            fields.invoice_date = _first_match(text, r"date\s*[:\-]\s*(\d{4}-\d{2}-\d{2})")
            fields.due_date = _first_match(text, r"due\s*date\s*[:\-]\s*(\d{4}-\d{2}-\d{2})")
            fields.po_number = _first_match(text, r"(?:po|p\.o\.)\s*#?\s*[:\-]?\s*(\S+)")
            total = _first_match(text, r"(?:grand\s*)?total\s*[:\-]?\s*\$?\s*([\d,]+\.?\d*)")
            fields.total_amount = _to_float(total)
            fields.grand_total = fields.total_amount
            fields.subtotal = _to_float(_first_match(text, r"subtotal\s*[:\-]?\s*\$?\s*([\d,]+\.?\d*)"))
            fields.tax_amount = _to_float(_first_match(text, r"tax\s*[:\-]?\s*\$?\s*([\d,]+\.?\d*)"))
            fields.vendor_tax_id = _first_match(text, r"(?:gstin|vat|ein|tax\s*id)\s*[:\-]?\s*(\S+)")
            fields.payment_terms = _first_match(text, r"(?:payment\s*)?terms\s*[:\-]\s*(.+)")
            fields.currency = "USD"
            confidence = 0.85 if fields.total_amount is not None else 0.5
            fields.line_items = _parse_line_items(text)
        elif document_type == DocumentType.CONTRACT:
            party = re.search(r"between\s+(.+?)\s+and\s+(.+?)[.\n]", text, re.IGNORECASE)
            if party:
                names = [party.group(1).strip(), party.group(2).strip()]
                fields.parties = names
                fields.party_names = names
            fields.effective_date = _first_match(text, r"effective\s+date\s*[:\-]\s*(\d{4}-\d{2}-\d{2})")
            fields.expiration_date = _first_match(text, r"(?:expir\w+|end)\s*date\s*[:\-]\s*(\d{4}-\d{2}-\d{2})")
            fields.governing_law = _first_match(text, r"governing\s+law\s*[:\-]\s*(.+)")
            fields.signatures_present = bool(re.search(r"signature|signed|/s/", text, re.IGNORECASE))
            confidence = 0.8 if fields.party_names else 0.5
        elif document_type == DocumentType.RESUME:
            fields.candidate_name = _first_match(text, r"name\s*[:\-]\s*(.+)")
            fields.full_name = fields.candidate_name
            fields.email = _first_match(text, r"[\w.\-]+@[\w.\-]+\.\w+")
            fields.phone = _first_match(text, r"(?:\+?\d{1,3}[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}")
            fields.location = _first_match(text, r"location\s*[:\-]\s*(.+)")
            fields.links = re.findall(r"https?://\S+", text)
            skills = _first_match(text, r"skills\s*[:\-]\s*(.+)")
            if skills:
                fields.skills = [s.strip() for s in re.split(r"[,;]", skills) if s.strip()]
            confidence = 0.8 if fields.email else 0.5
        elif document_type == DocumentType.ID_DOCUMENT:
            fields.full_name = _first_match(text, r"name\s*[:\-]\s*(.+)")
            fields.id_number = _first_match(text, r"id\s*(?:number|#)?\s*[:\-]?\s*(\S+)")
            fields.id_type = _first_match(text, r"(passport|driver'?s?\s*licen[cs]e|aadhaar|pan)")
            fields.date_of_birth = _first_match(text, r"(?:dob|date\s+of\s+birth)\s*[:\-]\s*(\d{4}-\d{2}-\d{2})")
            fields.expiry_date = _first_match(text, r"(?:expir\w+)\s*[:\-]?\s*(\d{4}-\d{2}-\d{2})")
            confidence = 0.8 if fields.id_number else 0.5

        return fields, round(confidence, 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count(text: str, keywords: list[str]) -> int:
    return sum(1 for kw in keywords if kw in text)


def _first_match(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip() if match.groups() else match.group(0).strip()


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _parse_line_items(text: str) -> list[LineItem]:
    """Parse simple ``qty x desc @ price = amount`` style lines."""
    items: list[LineItem] = []
    for match in re.finditer(
        r"(\d+)\s*x\s*(.+?)\s*@\s*\$?([\d.]+)\s*=\s*\$?([\d.]+)", text, flags=re.IGNORECASE
    ):
        qty, desc, unit, amount = match.groups()
        items.append(
            LineItem(
                description=desc.strip(),
                quantity=float(qty),
                unit_price=float(unit),
                amount=float(amount),
                line_total=float(amount),
            )
        )
    return items


def _safe_json_loads(raw: str) -> dict:
    """Parse JSON, tolerating code fences and conversational prefix/suffix text.

    Extracts the outermost ``{...}`` object so wrappers like a leading
    "Here is the JSON:" or markdown fences don't break parsing.
    """
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        raw = raw[start : end + 1]
    try:
        result = json.loads(raw)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        logger.warning("Could not parse LLM JSON response: %s", raw[:200])
        return {}


def _classification_from_payload(payload: dict) -> ClassificationResult:
    raw_type = str(payload.get("document_type", "unknown")).lower()
    try:
        doc_type = DocumentType(raw_type)
    except ValueError:
        doc_type = DocumentType.UNKNOWN
    return ClassificationResult(
        document_type=doc_type,
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        reasoning=str(payload.get("reasoning", "")),
    )


def _fields_from_payload(payload: dict) -> tuple[ExtractedFields, float]:
    confidence = float(payload.get("confidence", 0.7) or 0.7)
    # Drop the confidence key before validating against the field schema.
    field_payload = {k: v for k, v in payload.items() if k != "confidence"}
    try:
        fields = ExtractedFields.model_validate(field_payload)
    except Exception as exc:  # noqa: BLE001 - tolerate partial/extra fields
        logger.warning("Extraction payload validation failed: %s", exc)
        fields = ExtractedFields(extra={"raw": json.dumps(field_payload)[:500]})
    return fields, confidence


def get_llm_client(settings: Settings) -> LLMClient:
    """Return the appropriate LLM client based on settings."""
    if settings.use_mock_llm:
        logger.info("Using MockLLMClient (no Gemini calls will be made).")
        return MockLLMClient()
    logger.info("Using GeminiClient with model %s", settings.gemini_model)
    return GeminiClient(api_key=settings.gemini_api_key, model=settings.gemini_model)
