"""Small GitHub release API client shared by release maintenance scripts."""

import json
import os
import time
import urllib.error
import urllib.request


def token():
    return os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")


def request_json(url, *, authenticated=True, method="GET", payload=None):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "xllamacpp-release-tools",
    }
    if authenticated and token():
        headers["Authorization"] = f"Bearer {token()}"
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    for attempt in range(8):
        try:
            request = urllib.request.Request(url, data=data, headers=headers, method=method)
            with urllib.request.urlopen(request, timeout=90) as response:
                body = response.read()
                return json.loads(body) if body else None
        except urllib.error.HTTPError as error:
            if error.code not in (403, 429, 500, 502, 503, 504) or attempt == 7:
                raise
            if error.code == 403 and not (
                error.headers.get("Retry-After")
                or error.headers.get("X-RateLimit-Remaining") == "0"
                or b"rate limit" in error.read().lower()
            ):
                raise
            delay = max(min(60 * 2**attempt, 900), int(error.headers.get("Retry-After", "0")))
            if error.headers.get("X-RateLimit-Remaining") == "0":
                delay = max(delay, int(error.headers.get("X-RateLimit-Reset", "0")) - time.time() + 2)
            print(f"GitHub API retry after {delay:.0f}s (HTTP {error.code})", flush=True)
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError):
            if attempt == 7:
                raise
            time.sleep(10 * (attempt + 1))


def list_releases(repository, *, authenticated=True):
    releases = []
    page = 1
    while True:
        batch = request_json(
            f"https://api.github.com/repos/{repository}/releases?per_page=100&page={page}",
            authenticated=authenticated,
        )
        if not isinstance(batch, list):
            raise ValueError(f"Expected release list from {repository}")
        releases.extend(batch)
        if len(batch) < 100:
            return releases
        page += 1
