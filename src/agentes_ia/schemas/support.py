from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SupportQuery(BaseModel):
    """Validated classification of a support query."""

    model_config = ConfigDict(extra="forbid", strict=True)

    category: Literal["billing", "technical", "account", "other"]
    urgency: Literal["low", "medium", "high"]
    summary: str = Field(
        ..., min_length=1, max_length=200,
        description="A brief, single-sentence summary of the issue",
    )
    requires_escalation: bool
