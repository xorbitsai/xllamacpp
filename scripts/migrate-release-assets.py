#!/usr/bin/env python3
"""Resume copying release assets in independent, balanced GitHub Actions shards."""

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from github_releases import list_releases, request_json, token


def asset_matches(source, destination, digest=None):
    expected = digest or source.get("digest")
    return (
        destination.get("state") == "uploaded"
        and source["size"] == destination["size"]
        and (not expected or destination.get("digest") == expected)
    )


def partition(releases, count):
    groups = [[] for _ in range(count)]
    sizes = [0] * count
    for release in sorted(releases, key=lambda r: sum(a["size"] for a in r["assets"]), reverse=True):
        shard = min(range(count), key=lambda i: (sizes[i], len(groups[i])))
        groups[shard].append(release)
        sizes[shard] += sum(asset["size"] for asset in release["assets"])
    return groups


def upload_asset(release, asset, path, digest):
    parameters = {"name": asset["name"]}
    if asset.get("label"):
        parameters["label"] = asset["label"]
    url = release["upload_url"].split("{", 1)[0] + "?" + urllib.parse.urlencode(parameters)
    if urllib.parse.urlsplit(url).hostname != "uploads.github.com":
        raise ValueError("Unexpected upload host")
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token()}",
        "Content-Type": asset.get("content_type") or "application/octet-stream",
        "Content-Length": str(path.stat().st_size),
        "User-Agent": "xllamacpp-release-migration",
    }
    for attempt in range(10):
        try:
            with path.open("rb") as body:
                request = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=180) as response:
                    uploaded = json.load(response)
            if not asset_matches(asset, uploaded, digest):
                raise ValueError(f"Uploaded asset integrity mismatch: {asset['name']}")
            return uploaded
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            print(f"Upload attempt {attempt + 1} failed for {asset['name']}: {error}", flush=True)
            # A timed-out upload may already have succeeded. Never overwrite it blindly.
            existing = request_json(release["assets_url"] + "?per_page=100")
            for candidate in existing:
                if candidate["name"] != asset["name"]:
                    continue
                if asset_matches(asset, candidate, digest):
                    return candidate
                if candidate["state"] == "starter" and candidate["size"] == 0:
                    request_json(candidate["url"], method="DELETE")
                else:
                    raise ValueError(f"Conflicting destination asset: {asset['name']}") from error
            if attempt == 9:
                raise
            delay = min(60 * 2**attempt, 900)
            if isinstance(error, urllib.error.HTTPError):
                delay = max(delay, int(error.headers.get("Retry-After", "0")))
                if error.headers.get("X-RateLimit-Remaining") == "0":
                    delay = max(delay, int(error.headers.get("X-RateLimit-Reset", "0")) - time.time() + 2)
            time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="xorbitsai/xllamacpp-legacy")
    parser.add_argument("--destination", default="xorbitsai/xllamacpp")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--manifest", type=Path, default=Path(".github/release-migration-manifest.json"))
    args = parser.parse_args()
    if not token() or not 0 <= args.shard < args.shards:
        parser.error("A GitHub token and valid shard number are required")
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text())
        if manifest["source_repository"] != args.source or manifest["destination_repository"] != args.destination:
            raise ValueError("Migration manifest repositories do not match arguments")
        source = manifest["source"]
        destination = manifest["destination"]
    else:
        source = list_releases(args.source, authenticated=False)
        destination = {release["tag_name"]: release for release in list_releases(args.destination)}
    releases = partition(source, args.shards)[args.shard]
    report = []
    last_upload = 0
    for release in releases:
        tag = release["tag_name"]
        if tag not in destination:
            raise ValueError(f"Prepare destination release metadata before migrating assets: {tag}")
        target = destination[tag]
        existing = {asset["name"]: asset for asset in target["assets"]}
        for asset in release["assets"]:
            if asset["name"] in existing:
                candidate = existing[asset["name"]]
                if not asset_matches(asset, candidate):
                    raise ValueError(f"Conflicting destination asset: {tag}/{asset['name']}")
                print(f"Already copied: {tag}/{asset['name']}", flush=True)
                report.append({"tag": tag, "name": asset["name"], "size": asset["size"], "digest": candidate.get("digest"), "skipped": True})
                continue
            with tempfile.TemporaryDirectory(prefix="release-asset-") as temporary:
                path = Path(temporary) / "asset"
                # No credentials are sent to the public asset download URL or redirects.
                url = asset["browser_download_url"] + "?migration=" + str(int(time.time()))
                subprocess.run([
                    "curl", "--fail", "--location", "--silent", "--show-error",
                    "--retry", "5", "--retry-all-errors", "--connect-timeout", "30",
                    "--max-time", "900", "--output", str(path), "--", url,
                ], check=True)
                with path.open("rb") as body:
                    hasher = hashlib.sha256()
                    for chunk in iter(lambda: body.read(1024 * 1024), b""):
                        hasher.update(chunk)
                digest = "sha256:" + hasher.hexdigest()
                if path.stat().st_size != asset["size"] or (asset.get("digest") and asset["digest"] != digest):
                    raise ValueError(f"Source download integrity mismatch: {tag}/{asset['name']}")
                # Together the shards stay below GitHub's secondary write-rate limit.
                time.sleep(max(0, args.shards - (time.monotonic() - last_upload)))
                last_upload = time.monotonic()
                uploaded = upload_asset(target, asset, path, digest)
                report.append({"tag": tag, "name": asset["name"], "size": asset["size"], "digest": digest, "destination_digest": uploaded.get("digest")})
                print(f"Copied and verified: {tag}/{asset['name']} ({asset['size']} bytes)", flush=True)
            Path(f"migration-report-{args.shard}.json").write_text(json.dumps(report, indent=2) + "\n")
    Path(f"migration-report-{args.shard}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Shard {args.shard}: {len(report)} assets complete", flush=True)


if __name__ == "__main__":
    main()
