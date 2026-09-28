# 🛡️ Schema Guardian

> **Deterministic Structured Output & Schema Enforcement Microservice for Financial AI Pipelines.**

An asynchronous Python microservice demonstrating production-grade structured data extraction from unstructured financial news, filings, and transcripts using **Pydantic v2**, **`instructor`**, and **Google Gemini (`google-genai`)**.

---

## 🏗️ Architecture & Mechanics

```
┌─────────────────────────────────┐
│ Unstructured Financial Text     │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ SchemaGuardianExtractor                                 │
│                                                         │
│ 1. Generates OpenAPI schema via Pydantic v2             │
│ 2. Injects tool schema into Gemini API request          │
│ 3. Gemini generates tool call payload conforming to JSON│
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ Pydantic v2 Validation Gateway (Rust `pydantic-core`)   │
│                                                         │
│  - Field types: ticker (str), sentiment_score (float)   │
│  - Range constraints: [-1.0, 1.0], [0.0, 1.0]           │
│  - Custom @field_validator: uppercase NSE symbols       │
│  - Cross-field @model_validator: action-sentiment match │
└────────────────┬───────────────────────────────┬────────┘
                 │                               │
           [Valid Schema]                 [ValidationError]
                 │                               │
                 ▼                               ▼
       ┌───────────────────┐           ┌────────────────────────┐
       │ Strongly-Typed    │           │ Auto Self-Healing Loop │
       │ MarketCatalyst    │           │                        │
       │ Object Output     │           │ Feed validation trace  │
       └───────────────────┘           │ back to LLM context    │
                                       │ (max_retries=2)        │
                                       └────────────────────────┘
```

---

## 📁 Project Structure

```text
schema-guardian/
├── src/
│   ├── __init__.py
│   ├── models.py            # Pydantic v2 schemas (MarketCatalyst, CatalystReport)
│   └── extractor.py         # Async extraction engine + instructor Gemini client
├── tests/
│   ├── __init__.py
│   └── test_edge_cases.py   # Unit & live integration tests for boundary constraints
├── main.py                  # Interactive smoke test demo
├── requirements.txt         # Pinned production dependencies
├── .env.example             # Environment variable template
└── README.md                # Architectural documentation & interview guide
```

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Configure Gemini API key
cp .env.example .env
# Edit .env and insert your GEMINI_API_KEY
```

### 2. Run the Interactive Smoke Test
Executes single catalyst extraction (TCS earnings) and multi-event batch extraction (RELIANCE + INFY + Macro):
```bash
python main.py
```

### 3. Run the Test Suite
Executes 4 instant unit tests and 2 live LLM self-healing integration tests:
```bash
pytest tests/test_edge_cases.py -v
```

---

## 🧠 Key Pydantic v2 & `instructor` Implementation Details

### 1. Boundary & Cross-Field Enforcement
In [`src/models.py`](src/models.py):
- **`Field(ge=-1.0, le=1.0)`**: Hard numeric boundaries prevent sentiment hallucinations.
- **`@field_validator("ticker")`**: Strips whitespace, forces uppercase, and validates alphabetic characters or `'MACRO'`.
- **`@model_validator(mode="after")`**: Prevents logical contradictions by verifying that `LONG` recommendations have non-negative sentiment and `SHORT` recommendations have non-positive sentiment.

### 2. The Self-Healing Validation Loop
When an LLM violates a validation rule:
1. Pydantic raises a `ValidationError` with the exact parameter and failure explanation.
2. `instructor` intercepts the exception.
3. It appends the error trace to the conversation history as a user feedback turn:
   > *"The output had validation errors: `Contradictory extraction: recommended_action 'LONG' cannot have negative sentiment_score (-0.85)`"*
4. The model re-attempts generation with explicit correction guidance.

---

## 🎯 Senior AI Engineer / FDE Interview Cheat Sheet

### Q1: Why does asking an LLM for "raw JSON without markdown" fail in production?
- **Failure Modes:** Models hallucinate conversational preambles (*"Sure, here is the JSON:"*), wrap text in markdown fences (` ```json `), omit closing brackets under token limits, or invent non-existent enum values.
- **Solution:** Use provider-level function calling or constrained decoding (grammar masking) where tokens are constrained at the logit level.

### Q2: How does `instructor` wrap provider APIs under the hood?
1. Converts Pydantic models into OpenAPI JSON Schema via `model.model_json_schema()`.
2. Converts the schema into the provider's native tool definition format (`tools=[{"type": "function", ...}]`).
3. Sets `tool_choice` to enforce that the model *must* call this schema.
4. Deserializes the tool call payload back into the Pydantic model instance.
5. If validation fails, automatically executes the retry loop up to `max_retries`.

### Q3: How do you handle unrecoverable schema failures in production pipelines?
- Wrap extraction in a **Dead-Letter Queue (DLQ)** pattern: if retries exhaust (`max_retries=2`), catch `ValidationError`, quarantine the raw text and error payload into an audit queue (e.g. SQS / Kafka / Postgres DLQ), and trigger an observability alert (Langfuse / Datadog) rather than crashing the ingestion pipeline.
