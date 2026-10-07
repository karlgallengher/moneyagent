from __future__ import annotations

import tempfile
import unittest
from unittest.mock import Mock
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from enterprise_qa_agent.chat_qa_agent.src import memory_store
from scripts.web_auth import csrf_user, current_user
from enterprise_qa_agent.src.chat import catalog_router, domain_registry
from enterprise_qa_agent.src.chat import runner
from enterprise_qa_agent.src.mcp_tools import documents
from scripts.web_api import app


class UserDomainTests(unittest.TestCase):
    @staticmethod
    def slug(name: str) -> str:
        return f"u{'a' * 8}_{name}"

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patches = [
            patch.object(memory_store, "DB_PATH", self.root / "sessions.sqlite3"),
            patch.object(domain_registry, "REPO_ROOT", self.root),
            patch.object(domain_registry, "REGISTRY_PATH", self.root / "processed/user_domains.json"),
            patch.object(domain_registry, "UPLOAD_ROOT", self.root / "public_dataset_upload/raw_md"),
            patch.object(domain_registry, "INDEX_ROOT", self.root / "processed/user_page_indexes"),
            patch.object(documents, "REPO_ROOT", self.root),
            patch.object(documents, "UPLOAD_ROOT", self.root / "public_dataset_upload/raw_md"),
            patch.object(catalog_router, "REPO_ROOT", self.root),
        ]
        for context in self.patches:
            context.start()
        documents._catalog.cache_clear()
        documents._pages.cache_clear()
        documents._page_bm25.cache_clear()
        app.dependency_overrides[current_user] = lambda: {"user_id": "a" * 32}
        app.dependency_overrides[csrf_user] = lambda: {"user_id": "a" * 32}
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()
        for context in reversed(self.patches):
            context.stop()
        documents._catalog.cache_clear()
        documents._pages.cache_clear()
        documents._page_bm25.cache_clear()
        catalog_router.refresh_domain_paths()
        self.temp.cleanup()

    def test_create_upload_build_search_and_rebuild(self) -> None:
        created = self.client.post("/api/domains", json={
            "domain": "solar_bonds", "name": "绿色债券", "split_mode": "heading"
        }).json()
        self.assertTrue(created["ok"], created)
        slug = self.slug("solar_bonds")
        payload = "# 晨曦绿色债券\n\n本期绿色债券票面利率为 3.2%。"
        upload = self.client.post(
            f"/api/domains/{slug}/files",
            files={"file": ("bond.md", payload.encode("utf-8"), "text/markdown")},
        ).json()
        self.assertTrue(upload["ok"], upload)
        self.assertFalse(self.client.post(
            f"/api/domains/{slug}/files",
            files={"file": ("bond.md", b"# repeat", "text/markdown")},
        ).json()["ok"])
        built = self.client.post(f"/api/domains/{slug}/build").json()
        self.assertTrue(built["ok"], built)
        self.assertEqual(built["data"]["doc_count"], 1)
        old_path = domain_registry.domain_index_dirs()[slug]
        self.assertEqual(documents.search_pages("票面利率", slug)["results"][0]["doc_id"], "bond")
        self.assertTrue(any(
            hit["domain"] == slug
            for hit in catalog_router.cross_domain_catalog_search("晨曦绿色债券票面利率")
        ))
        listing = self.client.get("/api/domains").json()["data"]["domains"]
        self.assertTrue(next(row for row in listing if row["domain"] == slug)["page_index_exists"])

        new_doc = self.client.post(
            f"/api/domains/{slug}/files",
            files={"file": ("extra.md", "# 晨曦债券增补\n\n补充发行规模为五亿元。".encode("utf-8"), "text/markdown")},
        ).json()
        self.assertTrue(new_doc["ok"], new_doc)
        rebuilt = self.client.post(f"/api/domains/{slug}/build").json()
        self.assertTrue(rebuilt["ok"], rebuilt)
        self.assertEqual(rebuilt["data"]["doc_count"], 2)
        self.assertNotEqual(old_path, domain_registry.domain_index_dirs()[slug])
        self.assertEqual(documents.search_pages("发行规模", slug)["results"][0]["doc_id"], "extra")
        active_path = domain_registry.domain_index_dirs()[slug]
        with patch("scripts.build_page_index.build_index", side_effect=ValueError("build failed")):
            failed = self.client.post(f"/api/domains/{slug}/build").json()
        self.assertFalse(failed["ok"])
        self.assertEqual(domain_registry.domain_index_dirs()[slug], active_path)

    def test_invalid_file_and_unbuilt_domain(self) -> None:
        self.client.post("/api/domains", json={"domain": "new_notes", "name": "笔记"})
        slug = self.slug("new_notes")
        unbuilt = self.client.get("/api/domains").json()["data"]["domains"]
        self.assertFalse(next(row for row in unbuilt if row["domain"] == slug)["page_index_exists"])
        self.assertFalse(self.client.post(f"/api/domains/{slug}/build").json()["ok"])
        for filename, contents in [
            ("../unsafe.md", b"# invalid"),
            ("bad.pdf", b"%PDF"),
            ("bad.md", b"\xff"),
        ]:
            result = self.client.post(
                f"/api/domains/{slug}/files", files={"file": (filename, contents, "text/plain")}
            ).json()
            self.assertFalse(result["ok"], result)

    def test_manage_files_and_delete_domain(self) -> None:
        slug = self.slug("managed_notes")
        self.assertTrue(self.client.post("/api/domains", json={"domain": "managed_notes", "name": "管理测试"}).json()["ok"])
        original = "# 首次内容\n\n固定条款为第一版。".encode("utf-8")
        self.assertTrue(self.client.post(
            f"/api/domains/{slug}/files",
            files={"file": ("notes.md", original, "text/markdown")},
        ).json()["ok"])
        listing = self.client.get(f"/api/domains/{slug}/files").json()
        self.assertTrue(listing["ok"], listing)
        self.assertEqual(listing["data"]["files"][0]["filename"], "notes.md")
        self.assertFalse(self.client.put(
            f"/api/domains/{slug}/files/missing.md",
            files={"file": ("missing.md", b"# no", "text/markdown")},
        ).json()["ok"])
        self.assertTrue(self.client.post(f"/api/domains/{slug}/build").json()["ok"])
        active_index = domain_registry.domain_index_dirs()[slug]
        updated = "# 新内容\n\n固定条款为第二版。".encode("utf-8")
        replaced = self.client.put(
            f"/api/domains/{slug}/files/notes.md",
            files={"file": ("notes.md", updated, "text/markdown")},
        ).json()
        self.assertTrue(replaced["ok"], replaced)
        self.assertEqual((domain_registry.upload_dir(slug) / "notes.md").read_bytes(), updated)
        self.assertEqual(domain_registry.domain_index_dirs()[slug], active_index)
        details = next(item for item in self.client.get("/api/domains").json()["data"]["domains"] if item["domain"] == slug)
        self.assertEqual(details["status"], "needs_build")
        self.assertEqual(len(documents.search_pages("第一版", slug)["results"]), 1)
        self.assertTrue(self.client.post(f"/api/domains/{slug}/build").json()["ok"])
        self.assertEqual(next(
            item for item in self.client.get("/api/domains").json()["data"]["domains"] if item["domain"] == slug
        )["status"], "ready")
        removed_file = self.client.delete(f"/api/domains/{slug}/files/notes.md").json()
        self.assertTrue(removed_file["ok"], removed_file)
        self.assertFalse(self.client.get(f"/api/domains/{slug}/files").json()["data"]["files"])
        self.assertEqual(next(
            item for item in self.client.get("/api/domains").json()["data"]["domains"] if item["domain"] == slug
        )["status"], "needs_build")
        self.assertFalse(self.client.post(f"/api/domains/{slug}/build").json()["ok"])
        self.assertIn(slug, domain_registry.domain_index_dirs())
        self.assertEqual(self.client.delete("/api/domains/insurance").status_code, 404)
        removed_domain = self.client.delete(f"/api/domains/{slug}").json()
        self.assertTrue(removed_domain["ok"], removed_domain)
        self.assertNotIn(slug, domain_registry.load_domains())
        self.assertNotIn(slug, domain_registry.domain_index_dirs())
        self.assertFalse((domain_registry.UPLOAD_ROOT / slug).exists())
        self.assertFalse((domain_registry.INDEX_ROOT / slug).exists())

    def test_agent_loads_selected_domain_after_switch(self) -> None:
        for slug, title in [(self.slug("first_notes"), "文档甲"), (self.slug("second_notes"), "文档乙")]:
            self.assertTrue(self.client.post("/api/domains", json={"domain": slug[10:], "name": title}).json()["ok"])
            self.assertTrue(self.client.post(
                f"/api/domains/{slug}/files",
                files={"file": (f"{slug}.md", f"# {title}\n\n这是{title}的内容。".encode(), "text/markdown")},
            ).json()["ok"])
            self.assertTrue(self.client.post(f"/api/domains/{slug}/build").json()["ok"])

        paths = {
            domain: str((self.root / folder).resolve())
            for domain, folder in domain_registry.domain_index_dirs().items()
            if domain in {self.slug("first_notes"), self.slug("second_notes")}
        }
        graph = Mock()
        graph.invoke.return_value = {"final_answer": "stub", "status": "done", "token_usage": {}}
        with (
            patch.object(catalog_router, "domain_index_dirs", return_value=paths),
            patch("enterprise_qa_agent.src.graph.graph_builder.build_enterprise_graph", return_value=graph),
            patch("enterprise_qa_agent.src.io.output_utils.write_outputs"),
            patch.object(runner, "rerank_chat_docs_with_llm", side_effect=lambda query, question, agent: question),
        ):
            from enterprise_qa_agent.src import legacy_agent

            for slug in paths:
                result = runner.run_chat_query("文档内容是什么", requested_domain=slug)
                self.assertEqual(result["status"], "done")
                self.assertEqual(legacy_agent.PAGE_INDEX_PATH, f"{paths[slug]}/page_index.jsonl")
                self.assertEqual(legacy_agent.all_doc_ids_for_loaded_index(), [slug])
            auto = runner.run_chat_query("文档甲的内容是什么")
            self.assertEqual(auto["status"], "done")
            self.assertEqual(legacy_agent.PAGE_INDEX_PATH, f"{paths[self.slug('first_notes')]}/page_index.jsonl")


if __name__ == "__main__":
    unittest.main()
