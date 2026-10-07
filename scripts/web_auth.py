from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
import uuid

from fastapi import Depends, HTTPException, Request

from enterprise_qa_agent.chat_qa_agent.src.memory_store import connect_db, init_db
from enterprise_qa_agent.src.chat.domain_registry import claim_legacy_domains

COOKIE_NAME = "moneyagent_session"
SESSION_SECONDS = 7 * 24 * 60 * 60
USERNAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{2,31}$")
HASH_ROUNDS = 600_000


def init_auth_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS web_users (
            user_id TEXT PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS web_sessions (
            token_hash TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES web_users(user_id),
            csrf_token TEXT NOT NULL,
            expires_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS web_login_attempts (
            username TEXT PRIMARY KEY,
            attempts INTEGER NOT NULL,
            window_start INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS web_chat_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_web_chat_usage ON web_chat_usage(user_id, created_at);
        CREATE TABLE IF NOT EXISTS web_build_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_web_build_usage ON web_build_usage(user_id, created_at);
        """
    )


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, HASH_ROUNDS)
    return f"pbkdf2_sha256${HASH_ROUNDS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, rounds, salt, digest = stored.split("$")
        if algorithm != "pbkdf2_sha256" or int(rounds) < HASH_ROUNDS:
            return False
        computed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), int(rounds))
        return hmac.compare_digest(computed, bytes.fromhex(digest))
    except (ValueError, TypeError):
        return False


def create_user(conn: sqlite3.Connection, username: str, password: str, *, claim_legacy: bool = False) -> str:
    username = username.strip().lower()
    if not USERNAME_RE.fullmatch(username):
        raise ValueError("账号需为 3-32 位英文字母、数字、下划线或连字符，并以字母开头")
    if len(password) < 12 or len(password.encode("utf-8")) > 1024:
        raise ValueError("密码至少 12 个字符，且不能超过 1024 字节")
    init_db(conn)
    init_auth_db(conn)
    user_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO web_users(user_id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
        (user_id, username, hash_password(password), int(time.time())),
    )
    if claim_legacy:
        conn.execute("UPDATE sessions SET owner_id = ? WHERE owner_id IS NULL", (user_id,))
    conn.commit()
    if claim_legacy:
        claim_legacy_domains(user_id)
    return user_id


def authenticate(conn: sqlite3.Connection, username: str, password: str) -> dict | None:
    init_auth_db(conn)
    username = username.strip().lower()
    if not USERNAME_RE.fullmatch(username) or len(password.encode("utf-8")) > 1024:
        return None
    now = int(time.time())
    attempt = conn.execute(
        "SELECT attempts, window_start FROM web_login_attempts WHERE username = ?", (username,)
    ).fetchone()
    if attempt and now - attempt["window_start"] < 900 and attempt["attempts"] >= 5:
        return None
    row = conn.execute(
        "SELECT user_id, username, password_hash FROM web_users WHERE username = ?", (username,)
    ).fetchone()
    # Always run a password KDF, including for unknown accounts.
    fallback = "pbkdf2_sha256$600000$00000000000000000000000000000000$" + "00" * 32
    valid = verify_password(password, row["password_hash"] if row else fallback)
    if not row or not valid:
        conn.execute(
            """
            INSERT INTO web_login_attempts(username, attempts, window_start) VALUES (?, 1, ?)
            ON CONFLICT(username) DO UPDATE SET
              attempts = CASE WHEN ? - window_start >= 900 THEN 1 ELSE attempts + 1 END,
              window_start = CASE WHEN ? - window_start >= 900 THEN ? ELSE window_start END
            """,
            (username, now, now, now, now),
        )
        conn.commit()
        return None
    conn.execute("DELETE FROM web_login_attempts WHERE username = ?", (username,))
    conn.commit()
    return {"user_id": row["user_id"], "username": row["username"]}


def issue_session(conn: sqlite3.Connection, user_id: str) -> tuple[str, str]:
    now = int(time.time())
    token = secrets.token_urlsafe(32)
    csrf = secrets.token_urlsafe(32)
    conn.execute("DELETE FROM web_sessions WHERE expires_at <= ?", (now,))
    conn.execute(
        "INSERT INTO web_sessions(token_hash, user_id, csrf_token, expires_at) VALUES (?, ?, ?, ?)",
        (hashlib.sha256(token.encode()).hexdigest(), user_id, csrf, now + SESSION_SECONDS),
    )
    conn.commit()
    return token, csrf


def revoke_session(conn: sqlite3.Connection, token: str) -> None:
    conn.execute("DELETE FROM web_sessions WHERE token_hash = ?", (hashlib.sha256(token.encode()).hexdigest(),))
    conn.commit()


def current_user(request: Request) -> dict:
    token = request.cookies.get(COOKIE_NAME, "")
    if not token:
        raise HTTPException(status_code=401, detail="请先登录")
    with connect_db() as conn:
        init_auth_db(conn)
        row = conn.execute(
            """
            SELECT u.user_id, u.username, s.csrf_token
            FROM web_sessions s JOIN web_users u ON u.user_id = s.user_id
            WHERE s.token_hash = ? AND s.expires_at > ?
            """,
            (hashlib.sha256(token.encode()).hexdigest(), int(time.time())),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    return dict(row)


def csrf_user(request: Request, user: dict = Depends(current_user)) -> dict:
    header = request.headers.get("X-CSRF-Token", "")
    if not header or not hmac.compare_digest(header, user["csrf_token"]):
        raise HTTPException(status_code=403, detail="请求验证失败，请刷新页面后重试")
    return user


def require_session(conn: sqlite3.Connection, session_id: str, user_id: str) -> None:
    row = conn.execute(
        "SELECT 1 FROM sessions WHERE session_id = ? AND owner_id = ?", (session_id, user_id)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="会话不存在")


def consume_chat_quota(conn: sqlite3.Connection, user_id: str) -> None:
    limit = max(1, int(os.environ.get("MONEYAGENT_CHAT_PER_HOUR", "20")))
    now = int(time.time())
    conn.execute("DELETE FROM web_chat_usage WHERE created_at < ?", (now - 3600,))
    count = conn.execute(
        "SELECT COUNT(*) FROM web_chat_usage WHERE user_id = ? AND created_at >= ?",
        (user_id, now - 3600),
    ).fetchone()[0]
    if count >= limit:
        conn.commit()
        raise HTTPException(status_code=429, detail="本小时提问次数已用完，请稍后再试")
    conn.execute("INSERT INTO web_chat_usage(user_id, created_at) VALUES (?, ?)", (user_id, now))
    conn.commit()


def consume_build_quota(conn: sqlite3.Connection, user_id: str) -> None:
    init_auth_db(conn)
    limit = max(1, int(os.environ.get("MONEYAGENT_BUILDS_PER_HOUR", "5")))
    now = int(time.time())
    conn.execute("DELETE FROM web_build_usage WHERE created_at < ?", (now - 3600,))
    count = conn.execute(
        "SELECT COUNT(*) FROM web_build_usage WHERE user_id = ? AND created_at >= ?",
        (user_id, now - 3600),
    ).fetchone()[0]
    if count >= limit:
        conn.commit()
        raise HTTPException(status_code=429, detail="本小时构建次数已用完，请稍后再试")
    conn.execute("INSERT INTO web_build_usage(user_id, created_at) VALUES (?, ?)", (user_id, now))
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage invited MoneyAgent Web accounts.")
    parser.add_argument("username", help="Account name")
    parser.add_argument("--claim-legacy", action="store_true", help="Assign unowned sessions and domains to this account")
    args = parser.parse_args()
    password = getpass.getpass("New password (12+ characters): ")
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation:
        parser.error("passwords do not match")
    with connect_db() as conn:
        user_id = create_user(conn, args.username, password, claim_legacy=args.claim_legacy)
    print(f"Created account {args.username.lower()} (id={user_id}).")


if __name__ == "__main__":
    main()
