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
from io import BytesIO
from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path

from config import BASE_DIR
from services.resource_registry import Resource, ResourceRegistry, OPTIONAL_RESOURCES


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

    def _path(self, r): return self.base_dir / r.default_filename

    def load_dataframe(self, r):
        p = self._path(r)
        if not p.exists():
            if r.id in OPTIONAL_RESOURCES:
                return pd.DataFrame()
            raise FileNotFoundError(f"Resource {r.id} not found at {p}")
        return pd.read_csv(p)

    def save_dataframe(self, r, df): df.to_csv(self._path(r), index=False)

    def load_json(self, r):
        p = self._path(r)
        if not p.exists():
            if r.id in OPTIONAL_RESOURCES:
                return {}
            raise FileNotFoundError(f"Resource {r.id} not found at {p}")
        with open(p) as f: return json.load(f)

    def save_json(self, r, d):
        with open(self._path(r), 'w') as f: json.dump(d, f, indent=2)

    def load_pickle(self, r): return joblib.load(self._path(r))

    def save_pickle(self, r, o): joblib.dump(o, self._path(r))

    def exists(self, r): return self._path(r).exists()

    def delete(self, r): self._path(r).unlink(missing_ok=True)


class GitHubStorageProvider(StorageProvider):
    def __init__(self, owner, repo, branch, token):
        self.api = f"https://api.github.com/repos/{owner}/{repo}/contents"
        self.branch = branch
        self.token = token

    def _headers(self): return {"Authorization": f"token {self.token}", "Accept": "application/vnd.github.v3+json"}

    def _get_sha(self, r):
        url = f"{self.api}/{r.default_filename}?ref={self.branch}"
        resp = requests.get(url, headers=self._headers())
        return resp.json().get("sha") if resp.status_code == 200 else None

    def _crud(self, method, r, data=None):
        url = f"{self.api}/{r.default_filename}"
        sha = self._get_sha(r)
        payload = {"message": f"Update {r.id}", "branch": self.branch}
        if sha: payload["sha"] = sha
        if method == "put" and data: payload["content"] = base64.b64encode(data).decode()
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

    def load_dataframe(self, r):
        url = f"{self.api}/{r.default_filename}?ref={self.branch}"
        resp = requests.get(url, headers=self._headers())
        if resp.status_code == 404:
            if r.id in OPTIONAL_RESOURCES:
                return pd.DataFrame()
            raise FileNotFoundError(f"Resource {r.id} not found in GitHub")
        resp.raise_for_status()
        content = base64.b64decode(resp.json()["content"])
        return pd.read_csv(BytesIO(content)) if content.strip() else pd.DataFrame()

    def save_dataframe(self, r, df): self._crud("put", r, df.to_csv(index=False).encode())

    def load_json(self, r):
        url = f"{self.api}/{r.default_filename}?ref={self.branch}"
        resp = requests.get(url, headers=self._headers())
        if resp.status_code == 404:
            if r.id in OPTIONAL_RESOURCES:
                return {}
            raise FileNotFoundError(f"Resource {r.id} not found in GitHub")
        resp.raise_for_status()
        return json.loads(base64.b64decode(resp.json()["content"]))

    def save_json(self, r, d): self._crud("put", r, json.dumps(d, indent=2).encode())

    def load_pickle(self, r):
        url = f"{self.api}/{r.default_filename}?ref={self.branch}"
        resp = requests.get(url, headers=self._headers())
        if resp.status_code == 404: raise FileNotFoundError
        resp.raise_for_status()
        return joblib.load(BytesIO(base64.b64decode(resp.json()["content"])))

    def save_pickle(self, r, o):
        buf = BytesIO()
        joblib.dump(o, buf)
        self._crud("put", r, buf.getvalue())

    def exists(self, r):
        url = f"{self.api}/{r.default_filename}?ref={self.branch}"
        return requests.get(url, headers=self._headers()).status_code == 200

    def delete(self, r):
        sha = self._get_sha(r)
        if sha: requests.delete(f"{self.api}/{r.default_filename}", headers=self._headers(), json={"message":"delete","sha":sha,"branch":self.branch})


class DatabaseManager:
    def __init__(self, storage):
        self.storage = storage

    def load_history(self): return self.storage.load_dataframe(ResourceRegistry.HISTORY)
    def save_history(self, df): self.storage.save_dataframe(ResourceRegistry.HISTORY, df)
    def load_dataset(self): return self.storage.load_dataframe(ResourceRegistry.DATASET)
    def save_dataset(self, df): self.storage.save_dataframe(ResourceRegistry.DATASET, df)
    def load_dataset_with_goal(self): return self.storage.load_dataframe(ResourceRegistry.DATASET_WITH_GOAL)
    def save_dataset_with_goal(self, df): self.storage.save_dataframe(ResourceRegistry.DATASET_WITH_GOAL, df)
    def load_pending(self): return self.storage.load_dataframe(ResourceRegistry.PENDING)
    def save_pending(self, df): self.storage.save_dataframe(ResourceRegistry.PENDING, df)
    def load_model(self): return self.storage.load_pickle(ResourceRegistry.MODEL)
    def save_model(self, b): self.storage.save_pickle(ResourceRegistry.MODEL, b)
    def load_threshold(self): return self.storage.load_json(ResourceRegistry.THRESHOLD) if self.storage.exists(ResourceRegistry.THRESHOLD) else {}
    def save_threshold(self, d): self.storage.save_json(ResourceRegistry.THRESHOLD, d)
    def load_league_profile(self): return self.storage.load_dataframe(ResourceRegistry.LEAGUE_PROFILE) if self.storage.exists(ResourceRegistry.LEAGUE_PROFILE) else pd.DataFrame()
    def save_league_profile(self, df): self.storage.save_dataframe(ResourceRegistry.LEAGUE_PROFILE, df)
    def is_model_ready(self): return self.storage.exists(ResourceRegistry.MODEL)