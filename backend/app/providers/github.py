"""GitHub provider — the leading development signal.

For every subnet with a curated GitHub repo we pull recent commit/contributor/
release activity. ``GITHUB_TOKEN`` is optional but strongly recommended (60/h ->
5000/h). Subnets without a known repo simply get no dev contribution from here and
fall through to the demo provider during the merge step.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx

from ..registry import get_registry_entry
from .base import DevData, DevEvent

_API = "https://api.github.com"


def _normalise_repo(github: str | None) -> str | None:
    if not github:
        return None
    github = github.strip().rstrip("/")
    if github.count("/") == 1:
        return github
    return None  # org-only handles need a separate repo-discovery step


class GitHubProvider:
    name = "github"

    def __init__(self, token: str | None = None, timeout: float = 20.0) -> None:
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "alpha-intel"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(base_url=_API, timeout=timeout, headers=headers)

    def close(self) -> None:
        self._client.close()

    def _repo_dev(self, netuid: int, repo: str) -> DevData | None:
        since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        try:
            commits = self._client.get(
                f"/repos/{repo}/commits", params={"since": since, "per_page": 100}
            )
            if commits.status_code != 200:
                return None
            commit_rows = commits.json()
        except Exception:
            return None

        contributors: set[str] = set()
        last_commit_at: datetime | None = None
        events: list[DevEvent] = []
        for c in commit_rows:
            author = (c.get("author") or {}).get("login") or \
                     (c.get("commit", {}).get("author", {}) or {}).get("name", "")
            if author:
                contributors.add(author)
            msg = c.get("commit", {}).get("message", "").split("\n")[0]
            date_str = c.get("commit", {}).get("author", {}).get("date")
            when = None
            if date_str:
                try:
                    when = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    if last_commit_at is None or when > last_commit_at:
                        last_commit_at = when
                except ValueError:
                    pass
            if msg and len(events) < 3:
                events.append(DevEvent(
                    netuid=netuid, title=msg[:120],
                    detail=f"Commit in {repo}", source_url=c.get("html_url", ""),
                    occurred_at=when, raw_text=c.get("commit", {}).get("message", ""),
                ))

        releases_30d = 0
        try:
            rel = self._client.get(f"/repos/{repo}/releases", params={"per_page": 30})
            if rel.status_code == 200:
                cutoff = datetime.now(timezone.utc) - timedelta(days=30)
                for r in rel.json():
                    pub = r.get("published_at")
                    if pub:
                        try:
                            if datetime.fromisoformat(pub.replace("Z", "+00:00")) > cutoff:
                                releases_30d += 1
                        except ValueError:
                            pass
        except Exception:
            pass

        return DevData(
            netuid=netuid,
            commits_7d=len(commit_rows),
            contributors_7d=len(contributors),
            releases_30d=releases_30d,
            last_commit_at=last_commit_at,
            events=events,
        )

    def dev_data(self, netuids: list[int]) -> dict[int, DevData]:
        out: dict[int, DevData] = {}
        for n in netuids:
            repo = _normalise_repo(get_registry_entry(n).github)
            if not repo:
                continue
            dev = self._repo_dev(n, repo)
            if dev is not None:
                out[n] = dev
        if not out:
            raise RuntimeError("github provider produced no data")
        return out
