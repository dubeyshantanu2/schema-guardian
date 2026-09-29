"""Asynchronous extraction engine using instructor and Google Gemini GenAI SDK.

Wraps the Gemini client with instructor to enforce OpenAPI schema constraints
and handle automated self-healing validation retry loops.
"""

import os
import time
from typing import Optional, Tuple
from dotenv import load_dotenv
from google import genai
import instructor

from src.models import MarketCatalyst, CatalystReport, ExtractionMetrics

# Load environment variables from .env
load_dotenv()

SYSTEM_EXTRACTION_PROMPT = """You are Schema Guardian, an institutional quantitative analyst extracting structured catalysts from financial news.
Your role:
1. Extract deterministic, type-safe data strictly adhering to the schema.
2. For Indian equities, normalize company names to their National Stock Exchange (NSE) ticker symbol (e.g., 'TCS', 'RELIANCE', 'HDFCBANK'). If broader market, use 'MACRO'.
3. Calibrate sentiment scores between -1.0 (extremely bearish) and +1.0 (extremely bullish).
4. Ensure recommended actions align logically with sentiment polarity.
"""


class SchemaGuardianExtractor:
    """Async financial catalyst extraction engine with schema self-healing."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
    ):
        """Initialize the Gemini client and patch it with instructor.

        Args:
            api_key: Gemini API key. Defaults to GEMINI_API_KEY environment variable.
            model_name: Target Gemini model identifier (default: gemini-2.5-flash).
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set it in your environment or .env file."
            )

        self.model_name = model_name
        self.raw_client = genai.Client(api_key=self.api_key)
        
        # Patch client for async tool-calling via instructor
        self.client = instructor.from_genai(
            self.raw_client,
            use_async=True,
            mode=instructor.Mode.GENAI_TOOLS,
        )

    async def extract_catalyst(
        self,
        text: str,
        max_retries: int = 2,
    ) -> Tuple[MarketCatalyst, float]:
        """Extract a single MarketCatalyst from unstructured text with automatic retries.

        Args:
            text: Unstructured financial news article, filing, or transcript.
            max_retries: Number of self-healing retry attempts when validation errors occur.

        Returns:
            Tuple of (validated MarketCatalyst instance, extraction latency in seconds).
        """
        start_time = time.perf_counter()

        response: MarketCatalyst = await self.client.chat.completions.create(
            model=self.model_name,
            response_model=MarketCatalyst,
            max_retries=max_retries,
            messages=[
                {"role": "system", "content": SYSTEM_EXTRACTION_PROMPT},
                {
                    "role": "user",
                    "content": f"Extract market catalyst from following financial text:\n\n{text}",
                },
            ],
        )

        latency = time.perf_counter() - start_time
        return response, latency

    async def extract_report(
        self,
        text: str,
        max_retries: int = 2,
    ) -> Tuple[CatalystReport, float]:
        """Extract a multi-catalyst CatalystReport from comprehensive market texts.

        Args:
            text: Unstructured text covering one or more market movements.
            max_retries: Retry attempts on validation failure.

        Returns:
            Tuple of (validated CatalystReport instance, extraction latency in seconds).
        """
        start_time = time.perf_counter()

        response: CatalystReport = await self.client.chat.completions.create(
            model=self.model_name,
            response_model=CatalystReport,
            max_retries=max_retries,
            messages=[
                {"role": "system", "content": SYSTEM_EXTRACTION_PROMPT},
                {
                    "role": "user",
                    "content": f"Extract structured market report with all catalysts from text:\n\n{text}",
                },
            ],
        )

        latency = time.perf_counter() - start_time
        return response, latency

    async def extract_catalyst_with_metrics(
        self,
        text: str,
        max_retries: int = 2,
    ) -> Tuple[MarketCatalyst, ExtractionMetrics]:
        """Extract MarketCatalyst and collect detailed token telemetry and cost metrics."""
        start_time = time.perf_counter()

        response, raw_completion = await self.client.chat.completions.create_with_completion(
            model=self.model_name,
            response_model=MarketCatalyst,
            max_retries=max_retries,
            messages=[
                {"role": "system", "content": SYSTEM_EXTRACTION_PROMPT},
                {
                    "role": "user",
                    "content": f"Extract market catalyst from following financial text:\n\n{text}",
                },
            ],
        )

        latency = time.perf_counter() - start_time
        usage = getattr(raw_completion, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        candidates_tokens = getattr(usage, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage, "total_token_count", 0) or (prompt_tokens + candidates_tokens)

        # Gemini 2.5 Flash token pricing: $0.075 / 1M prompt tokens, $0.30 / 1M output tokens
        cost_usd = (prompt_tokens * 0.000000075) + (candidates_tokens * 0.00000030)

        metrics = ExtractionMetrics(
            latency_seconds=round(latency, 3),
            prompt_tokens=prompt_tokens,
            candidates_tokens=candidates_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=round(cost_usd, 7),
        )
        return response, metrics

    async def extract_report_with_metrics(
        self,
        text: str,
        max_retries: int = 2,
    ) -> Tuple[CatalystReport, ExtractionMetrics]:
        """Extract CatalystReport and collect detailed token telemetry and cost metrics."""
        start_time = time.perf_counter()

        response, raw_completion = await self.client.chat.completions.create_with_completion(
            model=self.model_name,
            response_model=CatalystReport,
            max_retries=max_retries,
            messages=[
                {"role": "system", "content": SYSTEM_EXTRACTION_PROMPT},
                {
                    "role": "user",
                    "content": f"Extract structured market report with all catalysts from text:\n\n{text}",
                },
            ],
        )

        latency = time.perf_counter() - start_time
        usage = getattr(raw_completion, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        candidates_tokens = getattr(usage, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage, "total_token_count", 0) or (prompt_tokens + candidates_tokens)

        cost_usd = (prompt_tokens * 0.000000075) + (candidates_tokens * 0.00000030)

        metrics = ExtractionMetrics(
            latency_seconds=round(latency, 3),
            prompt_tokens=prompt_tokens,
            candidates_tokens=candidates_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=round(cost_usd, 7),
        )
        return response, metrics

