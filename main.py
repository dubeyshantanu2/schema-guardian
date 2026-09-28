"""Smoke test and interactive demo runner for Schema Guardian."""

import asyncio
import json
from src.extractor import SchemaGuardianExtractor

SAMPLE_EARNINGS_ARTICLE = """
MUMBAI -- Tata Consultancy Services (TCS) reported a robust 8.2% year-on-year surge in Q3 net profit,
beating Dalal Street expectations led by strong momentum in BFSI and cloud migration deal wins.
Operating margins expanded by 65 bps to 25.1%. The board approved an interim dividend of Rs 10 per share.
CEO stated that enterprise AI transformation engagements pipeline has doubled, signaling accelerated tech spend.
Analysts recommend an aggressive BUY rating given the outperformance against peers.
"""

SAMPLE_MULTI_EVENT_ARTICLE = """
MUMBAI -- Indian equity markets experienced heightened volatility on Wednesday.
1. Reliance Industries (RELIANCE) saw aggressive buying following regulatory clearance for its new green energy gigafactory complex in Gujarat, with brokerages projecting significant EBITDA accretion by FY27.
2. Conversely, Infosys (INFY) shares witnessed heavy selling pressure after a whistleblower complaint alleged revenue recognition irregularities in its European BFSI delivery unit, prompting management to initiate an external audit.
3. On the macro front, the Reserve Bank of India maintained the repo rate unchanged at 6.5%, citing persistent food inflation concerns but retaining an accommodative medium-term growth projection.
"""


async def main() -> None:
    print("=" * 70)
    print("🚀 SCHEMA GUARDIAN: Structured Output & Schema Enforcement Demo")
    print("=" * 70)

    extractor = SchemaGuardianExtractor(model_name="gemini-2.5-flash")

    print("\n[1] Testing Single Catalyst Extraction (TCS Earnings):")
    catalyst, latency = await extractor.extract_catalyst(SAMPLE_EARNINGS_ARTICLE)
    print(f"⏱️ Extraction Latency: {latency:.2f}s")
    print("Validated Output (JSON):")
    print(json.dumps(catalyst.model_dump(), indent=2))

    print("\n" + "-" * 70)

    print("\n[2] Testing Multi-Event Batch Report (RELIANCE + INFY + MACRO):")
    report, report_latency = await extractor.extract_report(SAMPLE_MULTI_EVENT_ARTICLE)
    print(f"⏱️ Report Extraction Latency: {report_latency:.2f}s")
    print("Validated Report (JSON):")
    print(json.dumps(report.model_dump(), indent=2))

    print("\n" + "=" * 70)
    print("✅ All extractions satisfied 100% Pydantic v2 schema constraints.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
