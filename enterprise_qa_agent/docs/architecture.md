# Enterprise Agent Architecture

This project is a new enterprise QA / verification agent. It does not answer
competition questions directly.

## Input

The primary input is an atomic enterprise question:

```json
{
  "atomic_id": "res_b_009_C",
  "task_type": "claim_verification",
  "question": "请核验：...选项C的陈述是否成立：...",
  "entities": ["养殖", "消费电子制造", "光通信"],
  "metrics": [],
  "constraints": ["需要覆盖题干涉及的所有对象，不能只验证其中一部分。"]
}
```

## Target Output

The agent should return evidence-grounded enterprise results:

```json
{
  "atomic_id": "res_b_009_C",
  "status": "answered",
  "answer": "supported",
  "confidence": 0.82,
  "evidence": [
    {
      "doc_id": "pack2_text12",
      "chunk_id": "pack2_text12_lp_0003",
      "quote": "光模块技术约每四年完成一代迭代升级，同步实现单比特成本与功耗的同步减半。",
      "supports": "光通信领域同时存在成本/功耗下降与速率演进"
    }
  ],
  "reason": "简要说明证据如何支持或反驳该问题。"
}
```

## Graph

Recommended LangGraph flow:

```text
load_atomic_question
-> parse_question
-> route_documents
-> graph_retrieve
-> hybrid_retrieve
-> evidence_extract
-> verify_or_answer
-> final_result
```

## Node Responsibilities

### parse_question

Normalize the atomic question into structured slots:

- entities
- metrics
- relations
- constraints
- answer schema

### route_documents

Use entities, metrics, source domain, document titles, and graph index to select
candidate documents. This is not option-answer routing; it is enterprise
document routing.

### graph_retrieve

Use a lightweight GraphRAG index:

- `entity -> docs/sections/chunks`
- `metric -> chunks`
- `relation -> chunks`
- `entity - relation - entity/value -> evidence`

The first version can be a JSONL/SQLite index. A graph database is optional.

### hybrid_retrieve

Run BM25 and embedding retrieval inside candidate documents or graph-selected
sections. Merge and rerank candidates.

### evidence_extract

Compress long evidence blocks into short evidence facts:

```json
{
  "fact": "圣农发展通过自主育种降低养殖成本，料肉比下降可节约饲料成本约3亿元。",
  "evidence_id": "pack2_text16_lp_0021",
  "slot": "养殖领域降本机制"
}
```

### verify_or_answer

Answer based only on extracted evidence facts and source quotes:

- claim verification: `supported|contradicted|insufficient`
- rule applicability: `applicable|not_applicable|insufficient`
- calculation: calculated value plus formula
- extraction: extracted value
- ranking: ordered result

## Why Atomic Questions First

The old competition flow made the agent reason about question types and answer
letters. This new flow makes the agent reason about enterprise tasks:

- verify a statement
- extract a fact
- calculate a metric
- compare entities
- rank results
- check rule applicability

The A/B questions become a benchmark for enterprise QA behavior, not the runtime
interface.

