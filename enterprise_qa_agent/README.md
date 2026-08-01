# Enterprise QA Agent

This path is independent from the competition answering scripts.

The goal is to turn A/B benchmark questions into enterprise-style atomic
questions, then use those questions to build and evaluate a document QA /
claim verification agent.

## Pipeline

```text
A/B benchmark questions
-> atomic enterprise questions
-> document routing
-> hybrid retrieval / GraphRAG retrieval
-> evidence extraction
-> verification or answer generation
```

The output of this project is not `A/B/C/D` answers. It is structured,
evidence-grounded enterprise QA results.

## Atomic Question Types

- `claim_verification`: verify whether a statement is supported by documents.
- `fact_extraction`: extract a fact, name, date, amount, or text.
- `calculation`: calculate a value from document evidence.
- `comparison`: compare entities or metrics.
- `ranking`: produce an ordered result.
- `rule_applicability`: verify whether a rule applies to a situation.

## First Step

Generate atomic questions:

```powershell
D:\vscode\Projects\AFAC\venv\Scripts\python.exe D:\vscode\Projects\AFAC\enterprise_qa_agent\scripts\extract_atomic_questions.py
```

Outputs:

- `enterprise_qa_agent/data/atomic_questions/atomic_questions_a.jsonl`
- `enterprise_qa_agent/data/atomic_questions/atomic_questions_b.jsonl`
- `enterprise_qa_agent/data/atomic_questions/atomic_questions_all.jsonl`
- `enterprise_qa_agent/data/atomic_questions/summary.json`

## Old-Path Enterprise Agent Baseline

The old LangGraph path has been copied and adapted here:

```text
enterprise_qa_agent/scripts/enterprise_langgraph_agent.py
```

It reads one atomic question by `ATOMIC_ID`, adapts it to the old graph input
schema, runs retrieval/verification, and appends an enterprise-style JSONL
result.

Current runtime flow:

```text
ATOMIC_ID
-> load atomic question
-> infer domain
-> auto-select PAGE_INDEX_PATH unless explicitly overridden
-> prefilter candidate docs from document title / tree.md / first pages
-> run the copied old LangGraph path on those candidate docs
-> write enterprise JSONL result
```

Example:

```powershell
$env:ATOMIC_ID="res_b_009_C"
$env:QWEN_MODEL="deepseek-v4-flash"
D:\vscode\Projects\AFAC\venv\Scripts\python.exe D:\vscode\Projects\AFAC\enterprise_qa_agent\scripts\enterprise_langgraph_agent.py
```

You usually do not need to set `PAGE_INDEX_PATH`; it is inferred from the
atomic question domain. You can still override it manually. Supported defaults:

- `processed/page_index_insurance/page_index.jsonl`
- `processed/page_index_regulatory/page_index.jsonl`
- `processed/page_index_financial_reports/page_index.jsonl`
- `processed/page_index_financial_contracts/page_index.jsonl`
- `processed/page_index_research/page_index.jsonl`

Doc prefiltering can be adjusted with:

- `ENTERPRISE_DOC_TOP_K`, default `6`
- `DISABLE_ENTERPRISE_DOC_ROUTER=1`, use all docs in the selected domain

Outputs:

- debug logs: `enterprise_qa_agent/outputs/debug/`
- JSONL results: `enterprise_qa_agent/outputs/results/enterprise_results.jsonl`

## Mature-Path Chat QA Agent

The conversational QA entry reuses the mature old-path LangGraph rather than
the lightweight chat prototype. It adapts a user question into a free-form
legacy question, keeps the mature document routing, slot planning, evidence
search, audit, follow-up retrieval, and final reasoning flow.

Edit the question in code:

```text
enterprise_qa_agent/scripts/enterprise_chat_agent.py
```

Change:

```python
CHAT_QUERY = "..."
CHAT_DOMAIN = ""  # empty means auto route
```

Run:

```powershell
D:\vscode\Projects\AFAC\venv\Scripts\python.exe D:\vscode\Projects\AFAC\enterprise_qa_agent\scripts\enterprise_chat_agent.py
```

Outputs:

- debug logs: `enterprise_qa_agent/outputs/chat_debug/`
