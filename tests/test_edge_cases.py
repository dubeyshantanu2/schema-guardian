"""Edge-case test suite for Schema Guardian.

Validates schema constraints, field validation rules, cross-field consistency,
and self-healing retry behaviors against malformed and adversarial financial texts.
"""

import pytest
from pydantic import ValidationError
from src.models import MarketCatalyst
from src.extractor import SchemaGuardianExtractor


# =====================================================================
# Unit Tests: Pydantic v2 Schema Bounds & Validators (Instant / Local)
# =====================================================================

def test_ticker_field_validator_clean_and_reject() -> None:
    """Test that ticker validator strips whitespace, uppercases, and rejects non-alphabetic tickers."""
    # Valid normalized tickers
    cat1 = MarketCatalyst(
        ticker="  infy  ",
        sentiment_score=0.5,
        confidence=0.9,
        event_category="EARNINGS",
        key_drivers=["Revenue grew 10%"],
        recommended_action="LONG",
    )
    assert cat1.ticker == "INFY"

    # Valid 'MACRO' ticker
    cat_macro = MarketCatalyst(
        ticker="macro",
        sentiment_score=0.0,
        confidence=0.8,
        event_category="MACRO_POLICY",
        key_drivers=["Inflation steady"],
        recommended_action="STAND_ASIDE",
    )
    assert cat_macro.ticker == "MACRO"

    # Valid non-alphabetic NSE symbols with digits, ampersands, hyphens
    for raw_symbol, expected in [
        ("3mindia", "3MINDIA"),
        ("  m&m  ", "M&M"),
        ("bajaj-auto", "BAJAJ-AUTO"),
    ]:
        cat_special = MarketCatalyst(
            ticker=raw_symbol,
            sentiment_score=0.4,
            confidence=0.9,
            event_category="EARNINGS",
            key_drivers=["Strong revenue growth"],
            recommended_action="LONG",
        )
        assert cat_special.ticker == expected

    # Invalid ticker with prohibited punctuation must raise ValidationError
    with pytest.raises(ValidationError) as exc_info:
        MarketCatalyst(
            ticker="TCS_123!",
            sentiment_score=0.5,
            confidence=0.9,
            event_category="EARNINGS",
            key_drivers=["Strong margins"],
            recommended_action="LONG",
        )
    assert "Must be 'MACRO' or a valid uppercase NSE symbol" in str(exc_info.value)


def test_sentiment_score_bounds_enforced() -> None:
    """Ensure sentiment_score strictly rejects values outside [-1.0, 1.0]."""
    with pytest.raises(ValidationError) as exc_info:
        MarketCatalyst(
            ticker="RELIANCE",
            sentiment_score=1.5,  # Out of bounds (> 1.0)
            confidence=0.95,
            event_category="REGULATORY",
            key_drivers=["Expansion approved"],
            recommended_action="LONG",
        )
    assert "Input should be less than or equal to 1" in str(exc_info.value)


def test_key_drivers_length_constraint() -> None:
    """Ensure key_drivers strictly enforces between 1 and 4 items."""
    # Reject empty drivers list
    with pytest.raises(ValidationError):
        MarketCatalyst(
            ticker="TCS",
            sentiment_score=0.5,
            confidence=0.8,
            event_category="EARNINGS",
            key_drivers=[],  # min_length=1 violated
            recommended_action="LONG",
        )

    # Reject > 4 drivers
    with pytest.raises(ValidationError):
        MarketCatalyst(
            ticker="TCS",
            sentiment_score=0.5,
            confidence=0.8,
            event_category="EARNINGS",
            key_drivers=["Fact 1", "Fact 2", "Fact 3", "Fact 4", "Fact 5"],  # max_length=4 violated
            recommended_action="LONG",
        )


def test_cross_field_validator_action_sentiment_conflict() -> None:
    """Ensure cross-field @model_validator catches contradictory action and sentiment."""
    # Contradiction 1: LONG with negative sentiment
    with pytest.raises(ValidationError) as exc_info:
        MarketCatalyst(
            ticker="HDFCBANK",
            sentiment_score=-0.75,
            confidence=0.9,
            event_category="REGULATORY",
            key_drivers=["NPA ratio jumped 40 bps"],
            recommended_action="LONG",  # Contradiction!
        )
    assert "recommended_action 'LONG' cannot have negative sentiment_score" in str(exc_info.value)

    # Contradiction 2: SHORT with positive sentiment
    with pytest.raises(ValidationError) as exc_info_short:
        MarketCatalyst(
            ticker="HDFCBANK",
            sentiment_score=0.8,
            confidence=0.9,
            event_category="EARNINGS",
            key_drivers=["Record quarterly profit"],
            recommended_action="SHORT",  # Contradiction!
        )
    assert "recommended_action 'SHORT' cannot have positive sentiment_score" in str(exc_info_short.value)


# =====================================================================
# Live Integration Tests: LLM Extraction & Self-Healing Retries
# =====================================================================

@pytest.mark.asyncio
async def test_live_macro_extraction() -> None:
    """Verify live extractor maps market-wide news to ticker 'MACRO' and category 'MACRO_POLICY'."""
    extractor = SchemaGuardianExtractor(model_name="gemini-2.5-flash")
    raw_text = (
        "The Ministry of Finance announced an unexpected 0.5% GST hike on commercial vehicles, "
        "impacting domestic industry forecasts across multiple automakers."
    )
    catalyst, latency = await extractor.extract_catalyst(raw_text, max_retries=2)

    assert catalyst.ticker in ["MACRO", "TATAMOTORS", "MARUTI"]
    assert catalyst.event_category in ["REGULATORY", "MACRO_POLICY"]
    assert -1.0 <= catalyst.sentiment_score <= 0.0
    assert 1 <= len(catalyst.key_drivers) <= 4
    assert latency > 0.0


@pytest.mark.asyncio
async def test_live_contradictory_text_self_healing() -> None:
    """Verify instructor handles potentially confusing text and produces logically aligned output."""
    extractor = SchemaGuardianExtractor(model_name="gemini-2.5-flash")
    raw_text = """
    A prominent retail forum claimed: 'XYZ Bank is completely doomed, buy puts immediately!'
    However, HDFC Bank (HDFCBANK) officially clarified that its asset quality remains the best in industry,
    deposit franchise expanded 16%, and provisions declined to historic lows.
    """
    catalyst, latency = await extractor.extract_catalyst(raw_text, max_retries=2)

    assert catalyst.ticker == "HDFCBANK"
    # Action and sentiment must be aligned per model validator
    if catalyst.recommended_action == "LONG":
        assert catalyst.sentiment_score >= 0.0
    elif catalyst.recommended_action == "SHORT":
        assert catalyst.sentiment_score <= 0.0
