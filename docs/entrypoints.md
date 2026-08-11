# MoneyAgent Entrypoints

Use `scripts/run_moneyagent.py` as the unified entrypoint for daily work.

## Single-turn Chat

```powershell
python .\scripts\run_moneyagent.py chat -q "请比较平安e生保和太保团体百万医疗的免赔额规则" --domain insurance
```

## Multi-turn Session

Interactive session with SQLite memory:

```powershell
python .\scripts\run_moneyagent.py session --domain insurance
```

Reset a session:

```powershell
python .\scripts\run_moneyagent.py session --session-id default --reset
```

Run one turn through the session memory stack:

```powershell
python .\scripts\run_moneyagent.py session -q "上一轮提到的两个产品，哪个免赔额更宽松？" --domain insurance
```

## Evaluation

Run one case by case number:

```powershell
python .\scripts\run_moneyagent.py eval --case-index 1
```

Run one case by id:

```powershell
python .\scripts\run_moneyagent.py eval --case-id research_fr_001
```

Run all cases:

```powershell
python .\scripts\run_moneyagent.py eval --all
```

Evaluation outputs:

```text
enterprise_qa_agent/outputs/eval/moneyagent_eval_results.jsonl
enterprise_qa_agent/outputs/eval/moneyagent_eval_results_summary.json
enterprise_qa_agent/outputs/eval/moneyagent_eval_results_all_cases.jsonl
enterprise_qa_agent/outputs/eval/moneyagent_eval_results_all_cases_summary.json
```

## Legacy Entrypoints

These still work and are kept for compatibility:

```text
enterprise_qa_agent/scripts/enterprise_chat_agent.py
enterprise_qa_agent/scripts/enterprise_chat_session.py
scripts/evaluate_moneyagent.py
```
