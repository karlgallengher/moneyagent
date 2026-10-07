from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from collections.abc import Iterator

from .config import (
    COMPRESS_EVERY,
    DB_PATH,
    DELETE_COLD_EVIDENCE_AFTER_TURNS,
    MAX_EVIDENCE_MEMORY,
    MAX_EVIDENCE_SUMMARY_CHARS,
    MAX_FACTS_IN_CONTEXT,
    MAX_SUMMARY_CHARS,
    WINDOW_SIZE,
)
from .memory_policy import evidence_note_is_negative, infer_evidence_subject
from .utils import json_dumps, json_loads, now_iso, safe_text


@contextmanager
def connect_db() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '',
            active_domain TEXT NOT NULL DEFAULT '',
            active_doc_ids TEXT NOT NULL DEFAULT '[]',
            active_subjects TEXT NOT NULL DEFAULT '[]',
            global_summary TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            user_query TEXT NOT NULL,
            resolved_query TEXT NOT NULL,
            assistant_answer TEXT NOT NULL,
            domain TEXT NOT NULL DEFAULT '',
            doc_ids TEXT NOT NULL DEFAULT '[]',
            subjects TEXT NOT NULL DEFAULT '[]',
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            total_tokens INTEGER NOT NULL DEFAULT 0,
            debug_path TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        );

        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            subject TEXT NOT NULL DEFAULT '',
            fact TEXT NOT NULL,
            evidence_ids TEXT NOT NULL DEFAULT '[]',
            doc_ids TEXT NOT NULL DEFAULT '[]',
            source_turn INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        );

        CREATE TABLE IF NOT EXISTS evidence_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            evidence_id TEXT NOT NULL,
            doc_id TEXT NOT NULL DEFAULT '',
            subject TEXT NOT NULL DEFAULT '',
            slot TEXT NOT NULL DEFAULT '',
            evidence_summary TEXT NOT NULL DEFAULT '',
            evidence_status TEXT NOT NULL DEFAULT '',
            source_turn INTEGER NOT NULL,
            use_count INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            last_used_at TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id),
            UNIQUE(session_id, evidence_id, subject, slot)
        );
        """
    )
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)")}
    if "owner_id" not in columns:
        conn.execute("ALTER TABLE sessions ADD COLUMN owner_id TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_owner ON sessions(owner_id)")
    conn.commit()


def reset_session(conn: sqlite3.Connection, session_id: str) -> None:
    conn.execute("DELETE FROM evidence_memory WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM facts WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()


def ensure_session(conn: sqlite3.Connection, session_id: str) -> None:
    current = now_iso()
    conn.execute(
        """
        INSERT OR IGNORE INTO sessions(session_id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        (session_id, session_id, current, current),
    )
    conn.commit()


def list_sessions(conn: sqlite3.Connection, owner_id: str | None = None) -> list[dict]:
    where = "WHERE s.owner_id = ?" if owner_id is not None else ""
    rows = conn.execute(
        f"""
        SELECT
            s.session_id,
            s.title,
            s.active_domain,
            s.created_at,
            s.updated_at,
            COUNT(t.id) AS turn_count,
            MAX(t.created_at) AS last_turn_at
        FROM sessions s
        LEFT JOIN turns t ON t.session_id = s.session_id
        {where}
        GROUP BY s.session_id
        ORDER BY s.updated_at DESC, s.created_at DESC
        """,
        (owner_id,) if owner_id is not None else (),
    ).fetchall()
    return [
        {
            "session_id": str(row["session_id"]),
            "title": str(row["title"] or row["session_id"]),
            "active_domain": str(row["active_domain"] or ""),
            "turn_count": int(row["turn_count"] or 0),
            "created_at": str(row["created_at"] or ""),
            "updated_at": str(row["last_turn_at"] or row["updated_at"] or ""),
        }
        for row in rows
    ]


def update_session_title(conn: sqlite3.Connection, session_id: str, title: str) -> None:
    ensure_session(conn, session_id)
    clean_title = safe_text(title, 80).strip() or session_id
    conn.execute(
        "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
        (clean_title, now_iso(), session_id),
    )
    conn.commit()


def load_session_state(conn: sqlite3.Connection, session_id: str) -> dict:
    ensure_session(conn, session_id)
    session_row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    turn_rows = conn.execute(
        """
        SELECT * FROM turns
        WHERE session_id = ?
        ORDER BY turn_index DESC
        LIMIT ?
        """,
        (session_id, WINDOW_SIZE),
    ).fetchall()
    fact_rows = conn.execute(
        """
        SELECT * FROM facts
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (session_id, MAX_FACTS_IN_CONTEXT),
    ).fetchall()
    evidence_rows = conn.execute(
        """
        SELECT * FROM evidence_memory
        WHERE session_id = ? AND evidence_status IN ('useful', 'partial')
        ORDER BY last_used_at DESC, id DESC
        LIMIT ?
        """,
        (session_id, MAX_EVIDENCE_MEMORY * 3),
    ).fetchall()
    turn_count = conn.execute("SELECT COUNT(*) FROM turns WHERE session_id = ?", (session_id,)).fetchone()[0]
    return {
        "session_id": session_id,
        "turn_count": int(turn_count or 0),
        "active_domain": str(session_row["active_domain"] or ""),
        "active_doc_ids": json_loads(session_row["active_doc_ids"], []),
        "active_subjects": json_loads(session_row["active_subjects"], []),
        "global_summary": str(session_row["global_summary"] or ""),
        "turn_window": [dict(row) for row in reversed(turn_rows)],
        "facts": [dict(row) for row in reversed(fact_rows)],
        "evidence_memory": [dict(row) for row in reversed(evidence_rows)],
    }


def extract_subjects(final_state: dict) -> list[str]:
    subjects: list[str] = []
    question = final_state.get("question") or {}
    atomic = question.get("enterprise_atomic") or {}
    for value in atomic.get("entities") or []:
        text = safe_text(value, 40)
        if 2 <= len(text) <= 40 and text not in subjects:
            subjects.append(text)
    for task in final_state.get("task_reasoning_tasks") or []:
        for fact in task.get("known_facts") or []:
            slot = safe_text(fact.get("slot"), 80)
            value = safe_text(fact.get("value"), 80)
            if value and any(marker in slot for marker in ("主体", "对象", "领域", "行业", "公司", "产品", "案例")):
                if value not in subjects:
                    subjects.append(value)
    return subjects[:12]


def extract_facts(final_state: dict, max_facts: int = 12) -> list[dict]:
    facts: list[dict] = []
    for result in final_state.get("task_reasoning_results") or []:
        status = str(result.get("status") or "")
        if status not in {"complete", "partial"}:
            continue
        fact = safe_text(result.get("result") or result.get("basis"), 420)
        if not fact:
            continue
        facts.append(
            {
                "subject": safe_text(result.get("task") or result.get("slot"), 120),
                "fact": fact,
                "evidence_ids": list(result.get("evidence_ids") or [])[:6],
            }
        )
        if len(facts) >= max_facts:
            break
    return facts


def next_turn_index(conn: sqlite3.Connection, session_id: str) -> int:
    row = conn.execute("SELECT COALESCE(MAX(turn_index), 0) + 1 FROM turns WHERE session_id = ?", (session_id,)).fetchone()
    return int(row[0] or 1)


def extract_evidence_memories(final_state: dict, max_items: int = 20) -> list[dict]:
    memories: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()
    for result in final_state.get("task_reasoning_results") or []:
        if not isinstance(result, dict):
            continue
        status = str(result.get("status") or "")
        if status not in {"complete", "partial"}:
            continue
        evidence_status = "useful" if status == "complete" else "partial"
        slot = safe_text(result.get("task") or result.get("slot"), 120)
        for note in result.get("evidence_notes") or []:
            if not isinstance(note, dict):
                continue
            evidence_id = safe_text(note.get("source_evidence_id") or note.get("evidence_id"), 120)
            doc_id = safe_text(note.get("doc_id"), 80)
            note_text = safe_text(note.get("note"), MAX_EVIDENCE_SUMMARY_CHARS)
            if not evidence_id or not note_text or evidence_note_is_negative(note_text):
                continue
            subject = infer_evidence_subject(result, note)
            key = (evidence_id, doc_id, subject, slot)
            if key in seen:
                continue
            seen.add(key)
            memories.append(
                {
                    "evidence_id": evidence_id,
                    "doc_id": doc_id,
                    "subject": subject,
                    "slot": slot,
                    "evidence_summary": note_text,
                    "evidence_status": evidence_status,
                }
            )
            if len(memories) >= max_items:
                return memories
    return memories


def save_evidence_memories(conn: sqlite3.Connection, session_id: str, turn_index: int, final_state: dict, current: str) -> int:
    memories = extract_evidence_memories(final_state)
    for item in memories:
        conn.execute(
            """
            INSERT INTO evidence_memory(
                session_id, evidence_id, doc_id, subject, slot, evidence_summary,
                evidence_status, source_turn, use_count, created_at, last_used_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(session_id, evidence_id, subject, slot) DO UPDATE SET
                doc_id = excluded.doc_id,
                evidence_summary = excluded.evidence_summary,
                evidence_status = excluded.evidence_status,
                source_turn = excluded.source_turn,
                use_count = evidence_memory.use_count + 1,
                last_used_at = excluded.last_used_at
            """,
            (
                session_id,
                item["evidence_id"],
                item["doc_id"],
                item["subject"],
                item["slot"],
                item["evidence_summary"],
                item["evidence_status"],
                turn_index,
                current,
                current,
            ),
        )
    return len(memories)


def save_turn_and_memory(conn: sqlite3.Connection, session_id: str, query: str, resolved: dict, final_state: dict) -> int:
    question = final_state.get("question") or {}
    domain = str(question.get("enterprise_domain") or question.get("domain") or "")
    doc_ids = list(dict.fromkeys(str(doc_id) for doc_id in final_state.get("doc_ids") or question.get("doc_ids") or [] if doc_id))
    subjects = extract_subjects(final_state)
    facts = extract_facts(final_state)
    usage = final_state.get("token_usage") or {}
    turn_index = next_turn_index(conn, session_id)
    current = now_iso()

    conn.execute(
        """
        INSERT INTO turns(
            session_id, turn_index, user_query, resolved_query, assistant_answer,
            domain, doc_ids, subjects, prompt_tokens, completion_tokens, total_tokens, debug_path, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            turn_index,
            query,
            str(resolved.get("resolved_query") or ""),
            str(final_state.get("final_answer") or ""),
            domain,
            json_dumps(doc_ids),
            json_dumps(subjects),
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
            int(usage.get("total_tokens") or 0),
            str(final_state.get("debug_path") or ""),
            current,
        ),
    )

    for fact in facts:
        conn.execute(
            """
            INSERT INTO facts(session_id, subject, fact, evidence_ids, doc_ids, source_turn, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                str(fact.get("subject") or ""),
                str(fact.get("fact") or ""),
                json_dumps(fact.get("evidence_ids") or []),
                json_dumps(doc_ids),
                turn_index,
                current,
            ),
        )

    save_evidence_memories(conn, session_id, turn_index, final_state, current)

    conn.execute(
        """
        UPDATE sessions
        SET active_domain = ?,
            active_doc_ids = CASE WHEN ? != '[]' THEN ? ELSE active_doc_ids END,
            active_subjects = CASE WHEN ? != '[]' THEN ? ELSE active_subjects END,
            updated_at = ?
        WHERE session_id = ?
        """,
        (domain, json_dumps(doc_ids), json_dumps(doc_ids), json_dumps(subjects), json_dumps(subjects), current, session_id),
    )
    conn.commit()
    return turn_index


def save_local_turn(conn: sqlite3.Connection, session_id: str, query: str, answer: str, usage: dict | None = None) -> int:
    usage = usage or {}
    turn_index = next_turn_index(conn, session_id)
    current = now_iso()
    conn.execute(
        """
        INSERT INTO turns(
            session_id, turn_index, user_query, resolved_query, assistant_answer,
            domain, doc_ids, subjects, prompt_tokens, completion_tokens, total_tokens, debug_path, created_at
        )
        VALUES (?, ?, ?, ?, ?, '', '[]', '[]', ?, ?, ?, '', ?)
        """,
        (
            session_id,
            turn_index,
            query,
            query,
            answer,
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
            int(usage.get("total_tokens") or 0),
            current,
        ),
    )
    conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?", (current, session_id))
    conn.commit()
    return turn_index


def maybe_update_global_summary(conn: sqlite3.Connection, session_id: str, turn_index: int) -> None:
    if turn_index <= 0 or turn_index % COMPRESS_EVERY != 0:
        return
    session = conn.execute("SELECT global_summary FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    old_summary = str(session["global_summary"] or "") if session else ""
    rows = conn.execute(
        """
        SELECT turn_index, user_query, assistant_answer, domain, doc_ids
        FROM turns
        WHERE session_id = ? AND turn_index > ? AND turn_index <= ?
        ORDER BY turn_index ASC
        """,
        (session_id, max(0, turn_index - COMPRESS_EVERY), turn_index),
    ).fetchall()
    fragments = [old_summary] if old_summary else []
    for row in rows:
        answer = safe_text(row["assistant_answer"], 360)
        query = safe_text(row["user_query"], 160)
        doc_ids = ",".join(str(item) for item in json_loads(row["doc_ids"], [])[:6])
        fragments.append(f"Turn {row['turn_index']}: query={query}; domain={row['domain']}; docs={doc_ids}; answer={answer}")
    new_summary = safe_text(" ".join(fragment for fragment in fragments if fragment), MAX_SUMMARY_CHARS)
    conn.execute(
        "UPDATE sessions SET global_summary = ?, updated_at = ? WHERE session_id = ?",
        (new_summary, now_iso(), session_id),
    )
    conn.commit()


def mark_evidence_memory_used(conn: sqlite3.Connection, session_id: str, rows: list[dict]) -> None:
    ids = [int(row.get("id")) for row in rows if str(row.get("id") or "").isdigit()]
    if not ids:
        return
    current = now_iso()
    for memory_id in ids:
        conn.execute(
            """
            UPDATE evidence_memory
            SET use_count = use_count + 1,
                last_used_at = ?
            WHERE session_id = ? AND id = ?
            """,
            (current, session_id, memory_id),
        )
    conn.commit()


def cleanup_cold_evidence_memory(conn: sqlite3.Connection, session_id: str, current_turn_count: int) -> int:
    cutoff_turn = max(0, current_turn_count - DELETE_COLD_EVIDENCE_AFTER_TURNS)
    cursor = conn.execute(
        """
        DELETE FROM evidence_memory
        WHERE session_id = ?
          AND use_count <= 1
          AND source_turn <= ?
        """,
        (session_id, cutoff_turn),
    )
    deleted = int(cursor.rowcount or 0)
    if deleted:
        conn.commit()
    return deleted
