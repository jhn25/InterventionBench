"""HTTP client for amem_server (vector store backend).

Used by both MemGPT (archival_memory) and AMem (autonomous note-taking).
Wraps the FastAPI endpoints in scripts/amem_server.py.
"""
from __future__ import annotations
import os
import requests
from typing import Optional

AMEM_BASE = os.environ.get("AMEM_SERVER_URL", "http://localhost:9100")


class MemoryClient:
    """HTTP client for note storage + semantic retrieval per session."""

    def __init__(self, session_id: str, base_url: str = AMEM_BASE, timeout: float = 5.0):
        self.session_id = session_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._init_session()

    def _init_session(self):
        try:
            requests.post(f"{self.base_url}/init",
                          json={"session_id": self.session_id},
                          timeout=self.timeout)
        except Exception:
            pass  # server may be down; degrade gracefully

    def add(self, content: str, category: Optional[str] = None,
            tags: Optional[list[str]] = None) -> dict:
        try:
            r = requests.post(f"{self.base_url}/add",
                              json={"session_id": self.session_id,
                                    "content": content,
                                    "category": category,
                                    "tags": tags or []},
                              timeout=self.timeout)
            return r.json() if r.ok else {"ok": False, "error": str(r.status_code)}
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}

    def search(self, query: str, k: int = 5) -> list[dict]:
        try:
            r = requests.post(f"{self.base_url}/search",
                              json={"session_id": self.session_id,
                                    "query": query, "k": k},
                              timeout=self.timeout)
            if r.ok:
                resp = r.json()
                return resp.get("hits", []) if resp.get("ok") else []
            return []
        except Exception:
            return []
