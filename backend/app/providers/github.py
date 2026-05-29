"""GitHub provider — REAL development signal.

Pulls recent commit/contributor/release activity for each subnet's on-chain GitHub
repo (``SubnetIdentitiesV3.github_repo``). ``GITHUB_TOKEN`` is strongly recommended:
the anonymous limit (60 req/h) cannot cover ~100 repos, so without a token we scan a
rotating subset per cycle and leave the rest at zero (honest — never fabricated).

No synthetic data: a repo we couldn't read returns no DevData and the subnet simply
has no development signal that cycle.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone

import httpx

from .base import DevData, DevEvent

log = logging.getLogger("alpha.github")
_API = "https://api.github.com"
_REPO_RE = re.compile(r"github\.com[:/]+([^/]+/[^/#?]+?)(?:\.git)?/?$", re.IGNORECASE)


def parse_repo(github: str | None) -> str | None:
    """Extract ``owner/repo`` from a full GitHub URL or shorthand."""
    if not github:
        return None
    github = github.strip()
    m = _REPO_RE.search(github)
    if m:
        return m.group(1)
    if github.count("/") == 1 and " " not in github and "." not in github.split("/")[0]:
        return github
    return None


class GitHubProvider:
    name = "github"

    def __init__(self, token: str | None = None, timeout: float = 20.0,
                 anon_budget: int = 20) -> None:
        self.token = token
        self.anon_budget = anon_budget  # max repos to scan per cycle without a token
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "alpha-intel"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(base_url=_API, timeout=timeout, headers=headers,
                                    follow_redirects=True)

    def close(self) -> None:
        self._client.close()

    def _repo_dev(self, netuid: int, repo: str) -> DevData | None:
        since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        try:
            resp = self._client.get(
                f"/repos/{repo}/commits", params={"since": since, "per_page": 100}
            )
        except Exception:
            return None
        if resp.status_code != 200:
            return None
        commit_rows = resp.json()
        if not isinstance(commit_rows, list):
            return None

        contributors: set[str] = set()
        last_commit_at: datetime | None = None
        events: list[DevEvent] = []
        for c in commit_rows:
            commit = c.get("commit", {}) or {}
            author = (c.get("author") or {}).get("login") or \
                     (commit.get("author", {}) or {}).get("name", "")
            if author:
                contributors.add(author)
            msg = (commit.get("message") or "").split("\n")[0]
            date_str = (commit.get("author", {}) or {}).get("date")
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
                    netuid=netuid, title=msg[:140],
                    detail=f"Commit in {repo}", source_url=c.get("html_url", ""),
                    occurred_at=when, raw_text=msg,
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
            netuid=netuid, commits_7d=len(commit_rows),
            contributors_7d=len(contributors), releases_30d=releases_30d,
            last_commit_at=last_commit_at, events=events,
        )

    def dev_data_for_repos(self, repos: dict[int, str]) -> dict[int, DevData]:
        """``repos`` maps netuid -> github URL/shorthand (from on-chain identity)."""
        resolved = {n: r for n, g in repos.items() if (r := parse_repo(g))}
        order = sorted(resolved.keys())
        if not self.token:
            order = order[: self.anon_budget]
        out: dict[int, DevData] = {}
        for n in order:
            dev = self._repo_dev(n, resolved[n])
            if dev is not None:
                out[n] = dev
        log.info("github: dev data for %d/%d repos (token=%s)",
                 len(out), len(resolved), bool(self.token))
        return out
