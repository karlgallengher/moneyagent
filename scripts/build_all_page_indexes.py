from __future__ import annotations

from pathlib import Path

import build_page_index


REPO_ROOT = Path(__file__).resolve().parents[1]

DOMAIN_CONFIGS = [
    {
        "domain": "financial_contracts",
        "input": "public_dataset_upload/raw_md/financial_contracts",
        "output": "processed/page_index_financial_contracts",
        "split_mode": "heading",
    },
    {
        "domain": "financial_reports",
        "input": "public_dataset_upload/raw_md/financial_reports",
        "output": "processed/page_index_financial_reports",
        "split_mode": "heading",
    },
    {
        "domain": "insurance",
        "input": "public_dataset_upload/raw_md/insurance",
        "output": "processed/page_index_insurance",
        "split_mode": "heading_plus_numbered",
    },
    {
        "domain": "regulatory",
        "input": "public_dataset_upload/raw_md/regulatory",
        "output": "processed/page_index_regulatory",
        "split_mode": "heading_plus_numbered",
    },
    {
        "domain": "research",
        "input": "public_dataset_upload/raw_md/research",
        "output": "processed/page_index_research",
        "split_mode": "heading",
    },
]


def build_domain(config: dict) -> None:
    input_path = REPO_ROOT / config["input"]
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input for {config['domain']}: {input_path}")

    build_page_index.DEFAULT_INPUT = str(input_path)
    build_page_index.DEFAULT_OUTPUT = str(REPO_ROOT / config["output"])
    build_page_index.DEFAULT_SPLIT_MODE = str(config["split_mode"])
    build_page_index.DEFAULT_LIMIT = 0
    print(f"\n[build] domain={config['domain']}")
    print(f"[build] input={build_page_index.DEFAULT_INPUT}")
    print(f"[build] output={build_page_index.DEFAULT_OUTPUT}")
    build_page_index.main()


def main() -> None:
    for config in DOMAIN_CONFIGS:
        build_domain(config)


if __name__ == "__main__":
    main()
