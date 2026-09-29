"""Benchmarking suite for Schema Guardian.

Evaluates latency, token consumption, and dollar costs across realistic
financial news articles using Google Gemini 2.5 Flash and Pydantic v2.
"""

import asyncio
from typing import List, Dict, Any
from src.extractor import SchemaGuardianExtractor
from src.models import MarketCatalyst, CatalystReport, ExtractionMetrics


BENCHMARK_CORPUS = [
    {
        "id": "ART-01",
        "name": "TCS Q3 Earnings Beat",
        "type": "single",
        "text": """
        MUMBAI -- Tata Consultancy Services (TCS) reported an 8.2% year-on-year surge in Q3 net profit,
        beating Dalal Street expectations led by BFSI and cloud deal ramp-ups.
        Operating margins expanded 65 bps to 25.1%. The board declared an interim dividend of Rs 10 per share.
        Management noted double-digit growth in AI transformation contracts pipeline.
        """,
    },
    {
        "id": "ART-02",
        "name": "Tata Motors Demerger",
        "type": "single",
        "text": """
        MUMBAI -- Tata Motors (TATAMOTORS) board officially approved the composite scheme of arrangement
        to demerge its commercial vehicle and passenger vehicle (including EV and Jaguar Land Rover) businesses
        into two independent listed entities. The move aims to empower individual leadership teams
        and unlock shareholder value across distinct capital allocation strategies.
        """,
    },
    {
        "id": "ART-03",
        "name": "SEBI Algorithmic Trading Policy",
        "type": "single",
        "text": """
        NEW DELHI -- The Securities and Exchange Board of India (SEBI) issued comprehensive regulatory
        guidelines tightening margin requirements and testing protocols for algorithmic high-frequency trading.
        The policy introduces mandatory API throttle limits and stricter surveillance checks to safeguard retail
        participants from systemic flash crashes. Brokerages anticipate moderate short-term volume contractions.
        """,
    },
    {
        "id": "ART-04",
        "name": "Multi-Sector Volatility Round-up",
        "type": "report",
        "text": """
        MUMBAI -- Indian equity markets experienced heightened volatility on Wednesday.
        1. Reliance Industries (RELIANCE) gained 2.4% following environmental clearance for its gigafactory.
        2. Infosys (INFY) shares fell 3.1% amid management audits of a whistleblower allegation in Europe.
        3. The Reserve Bank of India maintained the benchmark repo rate at 6.5%, prioritizing food inflation management.
        """,
    },
]


async def run_benchmarks() -> None:
    print("=" * 88)
    print("⚡ SCHEMA GUARDIAN: LATENCY, TOKEN & COST PROFILING BENCHMARK")
    print("=" * 88)
    print("Target Model: gemini-2.5-flash")
    print("Pricing Model: $0.075 / 1M prompt tokens | $0.30 / 1M completion tokens\n")

    extractor = SchemaGuardianExtractor(model_name="gemini-2.5-flash")
    results: List[Dict[str, Any]] = []

    for article in BENCHMARK_CORPUS:
        print(f"Running [{article['id']}] {article['name']}...", end="", flush=True)

        if article["type"] == "single":
            catalyst, metrics = await extractor.extract_catalyst_with_metrics(article["text"])
            tickers = catalyst.ticker
            action = catalyst.recommended_action
        else:
            report, metrics = await extractor.extract_report_with_metrics(article["text"])
            tickers = ", ".join(c.ticker for c in report.catalysts)
            action = f"{len(report.catalysts)} catalysts"

        print(f" Done ({metrics.latency_seconds:.2f}s)")

        results.append({
            "id": article["id"],
            "name": article["name"][:28],
            "tickers": tickers,
            "action": action,
            "latency": metrics.latency_seconds,
            "prompt_tok": metrics.prompt_tokens,
            "cand_tok": metrics.candidates_tokens,
            "total_tok": metrics.total_tokens,
            "cost_usd": metrics.estimated_cost_usd,
        })

    # Print Formatted Results Table
    print("\n" + "-" * 88)
    header = (
        f"{'ID':<7} | {'Article Name':<28} | {'Ticker(s)':<14} | "
        f"{'Latency':<8} | {'Prompt':<7} | {'Compl':<6} | {'Total':<6} | {'Cost (USD)':<10}"
    )
    print(header)
    print("-" * 88)

    total_latency = 0.0
    total_prompt_tok = 0
    total_cand_tok = 0
    total_cost = 0.0

    for r in results:
        total_latency += r["latency"]
        total_prompt_tok += r["prompt_tok"]
        total_cand_tok += r["cand_tok"]
        total_cost += r["cost_usd"]

        print(
            f"{r['id']:<7} | {r['name']:<28} | {r['tickers']:<14} | "
            f"{r['latency']:>6.2f}s  | {r['prompt_tok']:>6}  | {r['cand_tok']:>5}  | "
            f"{r['total_tok']:>5}  | ${r['cost_usd']:>9.7f}"
        )

    print("-" * 88)
    n = len(results)
    avg_latency = total_latency / n
    total_tok = total_prompt_tok + total_cand_tok
    cost_per_1k_runs = (total_cost / n) * 1000

    print(f"\n📊 AGGREGATE TELEMETRY SUMMARY:")
    print(f"  • Total Runs Processed:      {n}")
    print(f"  • Average Latency:           {avg_latency:.2f} seconds")
    print(f"  • Total Tokens Consumed:     {total_tok:,} tokens ({total_prompt_tok:,} prompt + {total_cand_tok:,} completion)")
    print(f"  • Cumulative Incurred Cost:  ${total_cost:.7f} USD")
    print(f"  • Projected Cost / 1k Runs:  ${cost_per_1k_runs:.4f} USD")
    print("=" * 88)


if __name__ == "__main__":
    asyncio.run(run_benchmarks())
