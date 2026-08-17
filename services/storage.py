# services/storage.py
"""
Penyimpanan dan manajemen database untuk Football AI V2.
Mendukung penyimpanan lokal (filesystem) dan remote (GitHub API).
"""
import pandas as pd
import json
import os
import base64
import requests
import joblib
import time
from io import BytesIO
from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass
from pathlib import Path

from config import BASE_DIR
from services.resource_registry import Resource, ResourceRegistry, OPTIONAL_RESOURCES


@dataclass(frozen=True)
class DataFrameReadResult:
    """Hasil pembacaan DataFrame dengan status eksplisit agar error tidak dianggap data kosong."""
    status: str
    data: pd.DataFrame
    error: str | None = None


class StorageProvider(ABC):
    @abstractmethod
    def load_dataframe(self, resource: Resource) -> pd.DataFrame: ...
    @abstractmethod
    def save_dataframe(self, resource: Resource, df: pd.DataFrame): ...
    @abstractmethod
    def load_json(self, resource: Resource) -> dict: ...
    @abstractmethod
    def save_json(self, resource: Resource, data: dict): ...
    @abstractmethod
    def load_pickle(self, resource: Resource) -> Any: ...
    @abstractmethod
    def save_pickle(self, resource: Resource, obj: Any): ...
    @abstractmethod
    def exists(self, resource: Resource) -> bool: ...
    @abstractmethod
    def delete(self, resource: Resource): ...


class LocalStorageProvider(StorageProvider):
    def __init__(self, base_dir=BASE_DIR):
        self.base_dir = base_dir

    def _path(self, r):
        return self.base_dir / r.default_filename

    def load_dataframe(self, r):
        p = self._path(r)
        if not p.exists():
            if r.id in OPTIONAL_RESOURCES:
                return pd.DataFrame()
            raise FileNotFoundError(f"Resource {r.id} not found at {p}")
        return pd.read_csv(p)

    def save_dataframe(self, r, df):
        df.to_csv(self._path(r), index=False)

    def load_json(self, r):
        p = self._path(r)
        if not p.exists():
            if r.id in OPTIONAL_RESOURCES:
                return {}
            raise FileNotFoundError(f"Resource {r.id} not found at {p}")
        with open(p) as f:
            return json.load(f)

    def save_json(self, r, d):
        with open(self._path(r), 'w') as f:
            json.dump(d, f, indent=2)

    def load_pickle(self, r):
        return joblib.load(self._path(r))

    def save_pickle(self, r, o):
        joblib.dump(o, self._path(r))

    def exists(self, r):
        return self._path(r).exists()

    def delete(self, r):
        self._path(r).unlink(missing_ok=True)


class GitHubStorageProvider(StorageProvider):
    LARGE_FILE_THRESHOLD = 1_000_000  # 1 MB

    def __init__(self, owner, repo, branch, token):
        self.owner = owner
        self.repo = repo
        self.api = f"https://api.github.com/repos/{owner}/{repo}/contents"
        self.branch = branch
        self.token = token

    def _headers(self):
        return {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }

    @property
    def _repo_base_url(self):
        # https://api.github.com/repos/{owner}/{repo}
        return self.api.rsplit("/contents", 1)[0]

    def _get_sha(self, r):
        url = f"{self.api}/{r.default_filename}?ref={self.branch}"
        resp = requests.get(url, headers=self._headers())
        return resp.json().get("sha") if resp.status_code == 200 else None

    def _crud(self, method, r, data=None):
        url = f"{self.api}/{r.default_filename}"
        sha = self._get_sha(r)
        payload = {"message": f"Update {r.id}", "branch": self.branch}
        if sha:
            payload["sha"] = sha
        if method == "put" and data:
            payload["content"] = base64.b64encode(data).decode()
        resp = requests.request(method, url, headers=self._headers(), json=payload)
        if resp.status_code == 409:
            sha = self._get_sha(r)
            if sha:
                payload["sha"] = sha
                resp = requests.request(method, url, headers=self._headers(), json=payload)
                resp.raise_for_status()
            else:
                raise RuntimeError("Conflict: file tidak ditemukan setelah konflik")
        else:
            resp.raise_for_status()

    def _get_raw_content(self, r) -> bytes:
        """Ambil konten raw file dari repository private menggunakan Contents API.

        Menggunakan Accept: application/vnd.github.raw+json agar GitHub
        mengembalikan isi file mentah, bukan metadata base64.
        """
        url = f"{self.api}/{r.default_filename}?ref={self.branch}"
        headers = {
            **self._headers(),
            "Accept": "application/vnd.github.raw+json",
        }
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Gagal mengambil raw file {r.default_filename}: "
                f"HTTP {resp.status_code} - {resp.text[:300]}"
            )
        return resp.content

    def _save_large_file(self, r, data):
        """Panggil _do_save_large_file dengan retry sederhana."""
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                self._do_save_large_file(r, data)
                return
            except Exception as exc:
                if attempt == max_retries:
                    raise
                time.sleep(2 * attempt)

    def _do_save_large_file(self, r, data):
        """Simpan file besar menggunakan Git Data API (blob, tree, commit, ref)."""
        headers = self._headers()
        base_url = self._repo_base_url

        def _ensure_ok(resp, action):
            if resp.status_code not in (200, 201, 204):
                raise RuntimeError(
                    f"Git Data API error {resp.status_code} saat {action} "
                    f"untuk {r.default_filename}: {resp.text}"
                )

        # 1. Ambil SHA commit terakhir dari branch
        ref_url = f"{base_url}/git/refs/heads/{self.branch}"
        resp = requests.get(ref_url, headers=headers)
        _ensure_ok(resp, "GET ref")
        base_commit_sha = resp.json()["object"]["sha"]

        # 2. Ambil tree SHA dari commit terakhir
        commit_url = f"{base_url}/git/commits/{base_commit_sha}"
        resp = requests.get(commit_url, headers=headers)
        _ensure_ok(resp, "GET commit")
        base_tree_sha = resp.json()["tree"]["sha"]

        # 3. Buat blob baru
        blob_payload = {
            "content": base64.b64encode(data).decode(),
            "encoding": "base64",
        }
        resp = requests.post(f"{base_url}/git/blobs", headers=headers, json=blob_payload)
        _ensure_ok(resp, "POST blob")
        blob_sha = resp.json()["sha"]

        # 4. Buat tree baru
        tree_payload = {
            "base_tree": base_tree_sha,
            "tree": [
                {
                    "path": r.default_filename,
                    "mode": "100644",
                    "type": "blob",
                    "sha": blob_sha,
                }
            ],
        }
        resp = requests.post(f"{base_url}/git/trees", headers=headers, json=tree_payload)
        _ensure_ok(resp, "POST tree")
        new_tree_sha = resp.json()["sha"]

        # 5. Buat commit baru
        commit_payload = {
            "message": f"Update {r.id} via Git Data API",
            "tree": new_tree_sha,
            "parents": [base_commit_sha],
        }
        resp = requests.post(f"{base_url}/git/commits", headers=headers, json=commit_payload)
        _ensure_ok(resp, "POST commit")
        new_commit_sha = resp.json()["sha"]

        # 6. Update referensi branch
        ref_update_payload = {"sha": new_commit_sha, "force": False}
        resp = requests.patch(ref_url, headers=headers, json=ref_update_payload)
        _ensure_ok(resp, "PATCH ref")

    def load_dataframe(self, r):
        """Load CSV data from GitHub, with raw fallback for large files."""
        url = f"{self.api}/{r.default_filename}?ref={self.branch}"
        resp = requests.get(url, headers=self._headers())
        if resp.status_code == 404:
            if r.id in OPTIONAL_RESOURCES:
                return pd.DataFrame()
            raise FileNotFoundError(f"Resource {r.id} not found in GitHub")
        resp.raise_for_status()

        metadata = resp.json()
        content_b64 = metadata.get("content") or ""
        if content_b64.strip():
            content = base64.b64decode(content_b64)
            return pd.read_csv(BytesIO(content)) if content.strip() else pd.DataFrame()

        # Raw fallback untuk file besar
        content = self._get_raw_content(r)
        if not content.strip():
            return pd.DataFrame()
        return pd.read_csv(BytesIO(content))

    def save_dataframe(self, r, df):
        data = df.to_csv(index=False).encode()
        if len(data) > self.LARGE_FILE_THRESHOLD:
            try:
                self._save_large_file(r, data)
            except Exception as exc:
                raise RuntimeError(
                    f"Gagal menyimpan file besar {r.default_filename}: {exc}"
                )
        else:
            self._crud("put", r, data)

    def load_json(self, r):
        url = f"{self.api}/{r.default_filename}?ref={self.branch}"
        resp = requests.get(url, headers=self._headers())
        if resp.status_code == 404:
            if r.id in OPTIONAL_RESOURCES:
                return {}
            raise FileNotFoundError(f"Resource {r.id} not found in GitHub")
        resp.raise_for_status()

        metadata = resp.json()
        content_b64 = metadata.get("content") or ""
        if content_b64.strip():
            return json.loads(base64.b64decode(content_b64))

        # Raw fallback untuk file besar
        content = self._get_raw_content(r)
        return json.loads(content)

    def save_json(self, r, d):
        data = json.dumps(d, indent=2).encode()
        if len(data) > self.LARGE_FILE_THRESHOLD:
            try:
                self._save_large_file(r, data)
            except Exception as exc:
                raise RuntimeError(
                    f"Gagal menyimpan file besar {r.default_filename}: {exc}"
                )
        else:
            self._crud("put", r, data)

    def load_pickle(self, r):
        url = f"{self.api}/{r.default_filename}?ref={self.branch}"
        resp = requests.get(url, headers=self._headers())
        if resp.status_code == 404:
            raise FileNotFoundError
        resp.raise_for_status()

        metadata = resp.json()
        content_b64 = metadata.get("content") or ""
        if content_b64.strip():
            return joblib.load(BytesIO(base64.b64decode(content_b64)))

        # Raw fallback untuk file besar
        content = self._get_raw_content(r)
        return joblib.load(BytesIO(content))

    def save_pickle(self, r, o):
        buf = BytesIO()
        joblib.dump(o, buf)
        data = buf.getvalue()
        if len(data) > self.LARGE_FILE_THRESHOLD:
            try:
                self._save_large_file(r, data)
            except Exception as exc:
                raise RuntimeError(
                    f"Gagal menyimpan file besar {r.default_filename}: {exc}"
                )
        else:
            self._crud("put", r, data)

    def exists(self, r):
        url = f"{self.api}/{r.default_filename}?ref={self.branch}"
        return requests.get(url, headers=self._headers()).status_code == 200

    def delete(self, r):
        sha = self._get_sha(r)
        if sha:
            requests.delete(
                f"{self.api}/{r.default_filename}",
                headers=self._headers(),
                json={"message": "delete", "sha": sha, "branch": self.branch},
            )


class DatabaseManager:
    def __init__(self, storage):
        self.storage = storage

    def safe_load_dataframe(self, resource: Resource) -> DataFrameReadResult:
        """Bedakan resource hilang, benar-benar kosong, berhasil dibaca, dan error."""
        try:
            data = self.storage.load_dataframe(resource)
        except FileNotFoundError as exc:
            return DataFrameReadResult("MISSING", pd.DataFrame(), str(exc))
        except Exception as exc:
            return DataFrameReadResult("ERROR", pd.DataFrame(), str(exc))

        if data is None or data.empty:
            return DataFrameReadResult("EMPTY", pd.DataFrame() if data is None else data, None)
        return DataFrameReadResult("OK", data, None)

    def load_history(self):
        return self.storage.load_dataframe(ResourceRegistry.HISTORY)

    def save_history(self, df):
        self.storage.save_dataframe(ResourceRegistry.HISTORY, df)

    def load_dataset(self):
        return self.storage.load_dataframe(ResourceRegistry.DATASET)

    def save_dataset(self, df):
        self.storage.save_dataframe(ResourceRegistry.DATASET, df)

    def load_dataset_with_goal(self):
        return self.storage.load_dataframe(ResourceRegistry.DATASET_WITH_GOAL)

    def save_dataset_with_goal(self, df):
        self.storage.save_dataframe(ResourceRegistry.DATASET_WITH_GOAL, df)

    def load_pending(self):
        return self.storage.load_dataframe(ResourceRegistry.PENDING)

    def save_pending(self, df):
        self.storage.save_dataframe(ResourceRegistry.PENDING, df)

    def load_model(self):
        return self.storage.load_pickle(ResourceRegistry.MODEL)

    def save_model(self, b):
        self.storage.save_pickle(ResourceRegistry.MODEL, b)

    def load_threshold(self):
        return self.storage.load_json(ResourceRegistry.THRESHOLD) if self.storage.exists(ResourceRegistry.THRESHOLD) else {}

    def save_threshold(self, d):
        self.storage.save_json(ResourceRegistry.THRESHOLD, d)

    def load_league_profile(self):
        return self.storage.load_dataframe(ResourceRegistry.LEAGUE_PROFILE) if self.storage.exists(ResourceRegistry.LEAGUE_PROFILE) else pd.DataFrame()

    def save_league_profile(self, df):
        self.storage.save_dataframe(ResourceRegistry.LEAGUE_PROFILE, df)

    def is_model_ready(self):
        return self.storage.exists(ResourceRegistry.MODEL)