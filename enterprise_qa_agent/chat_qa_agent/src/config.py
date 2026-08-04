from __future__ import annotations

import os
import sys
from pathlib import Path


# Edit these values for session-level settings. Questions are typed after startup.
SESSION_ID = "default"
CHAT_DOMAIN = ""  # empty/auto, or one of: financial_contracts, financial_reports, insurance, regulatory, research
RESET_SESSION = False
CHAT_QUERY = ""  # optional one-shot mode

# Memory policy. Full turns stay in SQLite; only compact views are sent into the next query.
WINDOW_SIZE = 4
COMPRESS_EVERY = 4
MAX_SUMMARY_CHARS = 2500
MAX_FACTS_IN_CONTEXT = 12
MAX_EVIDENCE_MEMORY = 8
MAX_EVIDENCE_SUMMARY_CHARS = 220
HOT_EVIDENCE_TURN_WINDOW = 4
COLD_EVIDENCE_TURN_WINDOW = 12
DELETE_COLD_EVIDENCE_AFTER_TURNS = 50


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

SESSION_DIR = REPO_ROOT / "enterprise_qa_agent" / "outputs" / "chat_sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = SESSION_DIR / "chat_sessions.sqlite3"
