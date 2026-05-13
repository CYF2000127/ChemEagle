"""
ChemEagle Python client.

Stateless wrapper around the ``/api/v1`` HTTP endpoints. Supports both the
asynchronous (returns ``task_id``, you poll) and synchronous (``sync=True``)
modes, plus a convenience ``wait_for_task`` helper.

Example::

    from chemeagle_client import ChemEagleClient

    client = ChemEagleClient(
        base_url="https://app.chemeagle.net",
        api_key="ce_xxxxxxxxxxxx",
    )

    # Sync (blocks until done):
    result = client.process_image("scheme.png", sync=True)

    # Async (returns immediately, then poll):
    task_id = client.process_pdf("paper.pdf")["task_id"]
    final = client.wait_for_task(task_id)
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Union

import requests

PathLike = Union[str, os.PathLike]


class ChemEagleError(RuntimeError):
    """Raised when the API returns a non-success envelope."""

    def __init__(self, error_code: str, message: str, http_status: int,
                 payload: Optional[Dict[str, Any]] = None):
        super().__init__(f"{error_code}: {message} (HTTP {http_status})")
        self.error_code = error_code
        self.message = message
        self.http_status = http_status
        self.payload = payload or {}


class ChemEagleClient:
    """Thin HTTP client for the ChemEagle v1 API."""

    def __init__(self, base_url: str = "https://app.chemeagle.net",
                 api_key: Optional[str] = None,
                 timeout: float = 60.0,
                 session: Optional[requests.Session] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("CHEMEAGLE_API_KEY") or ""
        self.timeout = timeout
        self.session = session or requests.Session()

    # ----- low-level ----------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        h = {"Accept": "application/json"}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def _check(self, resp: requests.Response) -> Dict[str, Any]:
        try:
            payload = resp.json()
        except ValueError:
            raise ChemEagleError("internal_error",
                                 f"Non-JSON response (status {resp.status_code}): "
                                 f"{resp.text[:200]}",
                                 resp.status_code)
        if isinstance(payload, dict) and payload.get("success") is False:
            raise ChemEagleError(
                payload.get("error_code", "unknown"),
                payload.get("message", "unspecified"),
                resp.status_code,
                payload,
            )
        return payload

    # ----- endpoints ----------------------------------------------------
    def health(self) -> Dict[str, Any]:
        resp = self.session.get(self.base_url + "/api/v1/health",
                                headers=self._headers(),
                                timeout=self.timeout)
        return self._check(resp)

    def process_image(self, path: PathLike, *, sync: bool = False,
                      **extra_form: Any) -> Dict[str, Any]:
        """Upload a chemical-figure image. Returns task_id (or full result if sync)."""
        with open(path, "rb") as f:
            files = {"file": (os.path.basename(str(path)), f, "image/png")}
            data: Dict[str, Any] = {"sync": "true" if sync else "false"}
            data.update({k: str(v) for k, v in extra_form.items() if v is not None})
            resp = self.session.post(self.base_url + "/api/v1/process_image",
                                     headers=self._headers(),
                                     files=files, data=data,
                                     timeout=None if sync else self.timeout)
        return self._check(resp)

    def process_pdf(self, path: PathLike, *, sync: bool = False,
                    **extra_form: Any) -> Dict[str, Any]:
        """Upload a PDF for end-to-end extraction."""
        with open(path, "rb") as f:
            files = {"file": (os.path.basename(str(path)), f, "application/pdf")}
            data: Dict[str, Any] = {"sync": "true" if sync else "false"}
            data.update({k: str(v) for k, v in extra_form.items() if v is not None})
            resp = self.session.post(self.base_url + "/api/v1/process_pdf",
                                     headers=self._headers(),
                                     files=files, data=data,
                                     timeout=None if sync else self.timeout)
        return self._check(resp)

    def process_url(self, url: str, *, kind: str, sync: bool = False,
                    **extra: Any) -> Dict[str, Any]:
        """Have the server download an image or PDF from a URL, then dispatch."""
        body: Dict[str, Any] = {"url": url, "type": kind, "sync": sync}
        body.update(extra)
        resp = self.session.post(self.base_url + "/api/v1/process_url",
                                 headers={**self._headers(),
                                          "Content-Type": "application/json"},
                                 json=body,
                                 timeout=None if sync else self.timeout)
        return self._check(resp)

    def status(self, task_id: str) -> Dict[str, Any]:
        resp = self.session.get(f"{self.base_url}/api/v1/status/{task_id}",
                                headers=self._headers(),
                                timeout=self.timeout)
        return self._check(resp)

    def wait_for_task(self, task_id: str, *, poll: float = 2.0,
                      max_wait: float = 1800.0) -> Dict[str, Any]:
        """Block until the task completes / errors / max_wait expires."""
        deadline = time.time() + max_wait
        while time.time() < deadline:
            payload = self.status(task_id)
            status = payload.get("status")
            if status == "completed":
                return payload
            if status == "error":
                raise ChemEagleError(
                    payload.get("error_code", "internal_error"),
                    payload.get("message", "Task errored"),
                    500, payload,
                )
            time.sleep(poll)
        raise ChemEagleError("timeout",
                             f"Task {task_id} did not finish in {max_wait}s",
                             504)
