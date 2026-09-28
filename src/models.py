"""Data models and validation rules for market catalysts extraction.

Defines Pydantic v2 schemas used by instructor to generate OpenAPI tool
specifications for LLM structured output decoding and validation boundaries.
"""

from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class MarketCatalyst(BaseModel):
    """Structured representation of a market-moving event extracted from financial text.

    Constrained decoding ensures all fields adhere to enterprise database types,
    eliminating hallucinations and formatting anomalies before downstream persistence.
    """

    ticker: str = Field(
        description="NSE Ticker symbol in uppercase alphabetic characters (e.g. 'RELIANCE', 'TCS') or 'MACRO' if market-wide"
    )
    sentiment_score: float = Field(
        ge=-1.0,
        le=1.0,
        description="Polarity sentiment score ranging from -1.0 (extremely bearish) to +1.0 (extremely bullish)",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Extraction confidence score from 0.0 (uncertain) to 1.0 (high certainty)",
    )
    event_category: Literal[
        "REGULATORY", "EARNINGS", "MACRO_POLICY", "GEO_POLITICAL", "MANAGEMENT"
    ] = Field(
        description="Standardized classification of the market event"
    )
    key_drivers: list[str] = Field(
        min_length=1,
        max_length=4,
        description="Core supporting factual drivers extracted directly from text (1 to 4 concise bullets)",
    )
    recommended_action: Literal["LONG", "SHORT", "DELTA_NEUTRAL", "STAND_ASIDE"] = (
        Field(description="Algorithmic trading or allocation posture based on catalyst impact")
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Validate and normalize ticker format.

        Strips whitespace, converts to uppercase, and verifies that the string
        is either alphabetic or equals 'MACRO'.
        """
        clean = v.strip().upper()
        if not (clean.isalpha() or clean == "MACRO"):
            raise ValueError(
                f"Ticker '{v}' is invalid. Must be an alphabetic uppercase symbol (e.g. 'INFY') or 'MACRO'."
            )
        return clean

    @model_validator(mode="after")
    def validate_action_sentiment_consistency(self) -> "MarketCatalyst":
        """Cross-field validation ensuring recommended action matches sentiment polarity.

        Prevents contradictory LLM extractions (e.g. recommending LONG on a -0.8 bearish report).
        """
        if self.recommended_action == "LONG" and self.sentiment_score < 0.0:
            raise ValueError(
                f"Contradictory extraction: recommended_action 'LONG' cannot have negative sentiment_score ({self.sentiment_score})."
            )
        if self.recommended_action == "SHORT" and self.sentiment_score > 0.0:
            raise ValueError(
                f"Contradictory extraction: recommended_action 'SHORT' cannot have positive sentiment_score ({self.sentiment_score})."
            )
        return self


class CatalystReport(BaseModel):
    """Batch report container for documents covering multiple catalysts or market sectors."""

    headline: str = Field(description="Normalized summary headline of the source text")
    market_impact_summary: str = Field(
        description="Executive summary of overall market impact across mentioned assets"
    )
    catalysts: list[MarketCatalyst] = Field(
        min_length=1,
        description="List of one or more validated market catalyst entries",
    )
