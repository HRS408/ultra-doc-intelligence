"""Pydantic request/response models for Ultra Doc-Intelligence API."""

from typing import List, Optional

from pydantic import BaseModel


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    status: str = "indexed"
    doc_type: str = "general"
    load_id: str = ""


class SourceSnippet(BaseModel):
    text: str
    chunk_index: int
    similarity: float


class AskRequest(BaseModel):
    doc_id: str
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceSnippet]
    confidence: float
    confidence_label: str
    guardrail_triggered: bool
    guardrail_reason: Optional[str] = None


class ExtractRequest(BaseModel):
    doc_id: str


class ExtractResponse(BaseModel):
    shipment_id: Optional[str] = None
    shipper: Optional[str] = None
    consignee: Optional[str] = None
    pickup_datetime: Optional[str] = None
    delivery_datetime: Optional[str] = None
    equipment_type: Optional[str] = None
    mode: Optional[str] = None
    rate: Optional[float] = None
    currency: Optional[str] = None
    weight: Optional[float] = None
    carrier_name: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
