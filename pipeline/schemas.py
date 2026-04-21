from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

VALID_LABELS = [
    "core_content",
    "intro",
    "outro",
    "sponsorship",
    "self_promotion",
    "recap",
    "transition",
    "dead_air",
    "filler",
]

NON_CONTENT_LABELS = [l for l in VALID_LABELS if l != "core_content"]


class Segment(BaseModel):
    start: float
    end: float
    label: str
    type: Literal["content", "non-content"] = "content"
    confidence: float
    reason: str

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        if v not in VALID_LABELS:
            raise ValueError(f"Invalid label '{v}'. Must be one of {VALID_LABELS}")
        return v

    @model_validator(mode="after")
    def set_type_from_label(self):
        self.type = "content" if self.label == "core_content" else "non-content"
        return self


class AnalysisResult(BaseModel):
    video: str
    duration: float
    analyzed_at: str = ""
    segments: list[Segment]

    def model_post_init(self, __context):
        if not self.analyzed_at:
            self.analyzed_at = datetime.now(timezone.utc).isoformat()
