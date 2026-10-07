from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from enterprise_qa_agent.chat_qa_agent.src import memory_store
from enterprise_qa_agent.src.chat import catalog_router, domain_registry
from enterprise_qa_agent.src.mcp_tools import documents
from scripts.web_api import app
from scripts.web_auth import create_user


class WebAuthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.patches = [
            patch.object(memory_store, "DB_PATH", root / "sessions.sqlite3"),
            patch.object(domain_registry, "REPO_ROOT", root),
            patch.object(domain_registry, "REGISTRY_PATH", root / "processed/user_domains.json"),
            patch.object(domain_registry, "UPLOAD_ROOT", root / "uploads"),
            patch.object(domain_registry, "INDEX_ROOT", root / "indexes"),
            patch.object(documents, "REPO_ROOT", root),
            patch.object(documents, "UPLOAD_ROOT", root / "uploads"),
            patch.object(catalog_router, "REPO_ROOT", root),
        ]
        for item in self.patches:
            item.start()
        documents._catalog.cache_clear()
        documents._pages.cache_clear()
        documents._page_bm25.cache_clear()
        with memory_store.connect_db() as conn:
            self.alice_id = create_user(conn, "alice", "correct horse battery staple")
            self.bob_id = create_user(conn, "bob", "different secure passphrase")
        self.alice = self.login("alice", "correct horse battery staple")
        self.bob = self.login("bob", "different secure passphrase")

    def tearDown(self) -> None:
        for item in reversed(self.patches):
            item.stop()
        documents._catalog.cache_clear()
        documents._pages.cache_clear()
        documents._page_bm25.cache_clear()
        catalog_router.refresh_domain_paths()
        self.temp.cleanup()

    def login(self, username: str, password: str) -> TestClient:
        client = TestClient(app)
        response = client.post("/api/auth/login", json={"username": username, "password": password})
        self.assertEqual(response.status_code, 200, response.text)
        client.headers.update({"X-CSRF-Token": response.json()["data"]["csrf_token"]})
        return client

    def test_login_required_csrf_and_logout(self) -> None:
        stranger = TestClient(app)
        self.assertEqual(stranger.get("/api/sessions").status_code, 401)
        self.assertEqual(stranger.get("/api/domains").status_code, 401)
        self.assertEqual(stranger.post("/api/chat", json={"query": "hi"}).status_code, 401)
        self.assertEqual(stranger.post("/api/auth/login", json={
            "username": "alice", "password": "incorrect password",
        }).status_code, 401)
        self.assertEqual(self.alice.get("/api/auth/me").json()["data"]["username"], "alice")
        with self.alice as client:
            csrf = client.headers.pop("X-CSRF-Token")
            self.assertEqual(client.post("/api/sessions", json={}).status_code, 403)
            client.headers["X-CSRF-Token"] = csrf
            self.assertEqual(client.post("/api/auth/logout").status_code, 200)
            self.assertEqual(client.get("/api/sessions").status_code, 401)

    def test_sessions_and_chat_are_owner_scoped(self) -> None:
        created = self.alice.post("/api/sessions", json={"title": "私有研究"}).json()["data"]
        session_id = created["session_id"]
        self.assertEqual(len(self.alice.get("/api/sessions").json()["data"]["sessions"]), 1)
        self.assertEqual(self.bob.get("/api/sessions").json()["data"]["sessions"], [])
        self.assertEqual(self.bob.get(f"/api/sessions/{session_id}").status_code, 404)
        self.assertEqual(self.bob.patch(f"/api/sessions/{session_id}", json={"title": "盗用"}).status_code, 404)
        self.assertEqual(self.bob.delete(f"/api/sessions/{session_id}").status_code, 404)
        self.assertEqual(self.bob.post("/api/sessions/reset", json={"session_id": session_id}).status_code, 404)
        self.assertEqual(self.bob.post("/api/chat", json={"session_id": session_id, "query": "测试"}).status_code, 404)
        self.assertEqual(self.bob.post("/api/chat", json={"session_id": "guessed", "query": "测试"}).status_code, 404)
        self.assertEqual(self.alice.get(f"/api/sessions/{session_id}").status_code, 200)
        self.assertEqual(self.alice.get("/api/sessions").json()["data"]["sessions"][0]["title"], "私有研究")
        with memory_store.connect_db() as conn:
            self.assertIsNone(conn.execute("SELECT owner_id FROM sessions WHERE session_id = 'guessed'").fetchone())

    def test_domain_access_and_auto_search_are_owner_scoped(self) -> None:
        alice_domain = self.alice.post("/api/domains", json={"domain": "notes", "name": "甲资料"}).json()["data"]["domain"]
        bob_domain = self.bob.post("/api/domains", json={"domain": "notes", "name": "乙资料"}).json()["data"]["domain"]
        self.assertNotEqual(alice_domain, bob_domain)
        self.assertEqual(self.alice.post(
            f"/api/domains/{alice_domain}/files",
            files={"file": ("alice.md", "# 独有研究\n\n甲资料的独有研究。".encode(), "text/markdown")},
        ).status_code, 200)
        self.assertTrue(self.alice.post(f"/api/domains/{alice_domain}/build").json()["ok"])
        self.assertTrue(self.bob.post(
            f"/api/domains/{bob_domain}/files",
            files={"file": ("bob.md", "# 乙报告\n\n乙报告的金融研究。".encode(), "text/markdown")},
        ).json()["ok"])
        self.assertTrue(self.bob.post(f"/api/domains/{bob_domain}/build").json()["ok"])

        self.assertNotIn(bob_domain, [item["domain"] for item in self.alice.get("/api/domains").json()["data"]["domains"]])
        self.assertEqual(self.alice.get(f"/api/domains/{bob_domain}/files").status_code, 404)
        self.assertEqual(self.alice.post(f"/api/domains/{bob_domain}/build").status_code, 404)
        self.assertEqual(self.alice.delete(f"/api/domains/{bob_domain}").status_code, 404)
        self.assertEqual(self.alice.post(
            "/api/chat", json={"session_id": "missing", "query": "测试", "domain": bob_domain}
        ).status_code, 404)

        session = self.alice.post("/api/sessions", json={}).json()["data"]["session_id"]
        captured = []

        def inspect_route(*args):
            visible = domain_registry.domain_index_dirs()
            hits = catalog_router.cross_domain_catalog_search("乙报告的金融研究")
            captured.append((visible, hits))
            return {"final_answer": "ok", "token_usage": {}, "question": {}}

        response = Mock()
        response.to_dict.return_value = {"answer": "ok"}
        with (
            patch("scripts.web_api.run_one_turn", side_effect=inspect_route),
            patch("scripts.web_api.state_to_response", return_value=response),
        ):
            self.assertTrue(self.alice.post(
                "/api/chat", json={"session_id": session, "query": "乙报告的金融研究"}
            ).json()["ok"])
        self.assertIn(alice_domain, captured[0][0])
        self.assertNotIn(bob_domain, captured[0][0])
        self.assertNotIn(bob_domain, [hit["domain"] for hit in captured[0][1]])

    def test_legacy_data_requires_explicit_claim(self) -> None:
        with memory_store.connect_db() as conn:
            memory_store.ensure_session(conn, "legacy-one")
        domain_registry.create_domain("legacy_notes", "旧资料")
        self.assertEqual(self.alice.get("/api/sessions").json()["data"]["sessions"], [])
        self.assertNotIn("legacy_notes", [item["domain"] for item in self.alice.get("/api/domains").json()["data"]["domains"]])
        with memory_store.connect_db() as conn:
            create_user(conn, "owner", "another long secret password", claim_legacy=True)
        owner = self.login("owner", "another long secret password")
        self.assertEqual(owner.get("/api/sessions").json()["data"]["sessions"][0]["session_id"], "legacy-one")
        self.assertIn("legacy_notes", [item["domain"] for item in owner.get("/api/domains").json()["data"]["domains"]])
        self.assertEqual(self.bob.get("/api/sessions").json()["data"]["sessions"], [])

    def test_user_quotas(self) -> None:
        with patch.dict("os.environ", {"MONEYAGENT_DOMAINS_PER_USER": "1", "MONEYAGENT_FILES_PER_DOMAIN": "1",
                                      "MONEYAGENT_BUILDS_PER_HOUR": "1", "MONEYAGENT_CHAT_PER_HOUR": "1"}):
            domain = self.alice.post("/api/domains", json={"domain": "one", "name": "甲资料"}).json()["data"]["domain"]
            self.assertEqual(self.alice.post("/api/domains", json={"domain": "two", "name": "甲资料二"}).status_code, 429)
            self.assertTrue(self.alice.post(
                f"/api/domains/{domain}/files",
                files={"file": ("first.md", "# 一\n\n文件内容".encode(), "text/markdown")},
            ).json()["ok"])
            self.assertEqual(self.alice.post(
                f"/api/domains/{domain}/files",
                files={"file": ("second.md", b"# another", "text/markdown")},
            ).status_code, 429)
            self.assertTrue(self.alice.post(f"/api/domains/{domain}/build").json()["ok"])
            self.assertEqual(self.alice.post(f"/api/domains/{domain}/build").status_code, 429)

            session = self.alice.post("/api/sessions", json={}).json()["data"]["session_id"]
            response = Mock()
            response.to_dict.return_value = {"answer": "ok"}
            with (
                patch("scripts.web_api.run_one_turn", return_value={"final_answer": "ok", "token_usage": {}}),
                patch("scripts.web_api.state_to_response", return_value=response),
            ):
                self.assertTrue(self.alice.post(
                    "/api/chat", json={"session_id": session, "query": "第一问"}
                ).json()["ok"])
                self.assertEqual(self.alice.post(
                    "/api/chat", json={"session_id": session, "query": "第二问"}
                ).status_code, 429)


if __name__ == "__main__":
    unittest.main()
