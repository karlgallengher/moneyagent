# MoneyAgent MCP Tools

MoneyAgent exposes a small MCP tool layer for document research and deterministic calculations. The current design exposes tools first, not the full agent workflow.

## Tools

```text
list_domains_tool
```

List all available domains and index health.

```text
search_docs_tool
```

Search document catalogs in one domain.

```text
search_pages_tool
```

Search page-level evidence in one domain, optionally limited by `doc_ids`.

```text
get_page_tool
```

Fetch one indexed page by `page_id` or `doc_id`.

```text
get_doc_outline_tool
```

Fetch one document's metadata and heading outline.

```text
calculate_finance_tool
```

Run deterministic calculations:

```text
difference
sum
ratio
yoy_reverse
bond_interest
deductible_claim
```

```text
ask_moneyagent_tool
```

Ask the full MoneyAgent workflow and return the final answer, selected domain, document ids, evidence, and token usage.

## Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start the MCP server:

```powershell
python .\scripts\mcp_server.py
```

Or start it through the unified entrypoint:

```powershell
python .\scripts\run_moneyagent.py mcp
```

The server uses stdio transport through MCP, so it is meant to be launched by an MCP-compatible client.

Smoke test the MCP tool layer without starting a stdio server:

```powershell
python .\scripts\mcp_smoke_test.py
```

## Example Tool Inputs

Search documents:

```json
{
  "query": "平安e生保 家庭共享免赔额",
  "domain": "insurance",
  "top_k": 5
}
```

Search pages:

```json
{
  "query": "家庭共享免赔额1万元",
  "domain": "insurance",
  "doc_ids": ["5"],
  "top_k": 3
}
```

Calculate bond interest:

```json
{
  "operation": "bond_interest",
  "params": {
    "principal": 10000,
    "rate": 0.015,
    "days": 120
  }
}
```

Ask MoneyAgent:

```json
{
  "query": "平安e生保的家庭共享免赔额规则是什么？",
  "domain": "insurance",
  "preferred_doc_ids": ["5"],
  "include_raw_state": false
}
```
