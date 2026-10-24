#!/usr/bin/env python3
"""Build PEP 503 wheel indexes, preferring the active repository's assets."""

import argparse
import html
import re
from pathlib import Path

from github_releases import list_releases


def channel_for_tag(tag):
    match = re.fullmatch(r"v?(\d+\.\d+\.\d+(?:\.\d+)?)(?:-(.+))?", tag)
    if not match:
        return None
    version, suffix = match.groups()
    if suffix is None:
        channel = "cpu"
    elif suffix.startswith("vulkan-"):
        channel = "vulkan"
    elif re.fullmatch(r"cu\d+|rocm-\d+(?:\.\d+)+", suffix):
        channel = suffix
    else:
        return None
    return channel, tuple(int(part) for part in version.split("."))


def collect_wheels(release_groups):
    """Later groups override matching wheel filenames in earlier groups."""
    wheels = {}
    for releases in release_groups:
        for release in releases:
            parsed = channel_for_tag(release["tag_name"])
            if release.get("draft") or parsed is None:
                continue
            channel, version = parsed
            for asset in release["assets"]:
                name = asset["name"]
                if not name.endswith(".whl") or asset.get("state") != "uploaded" or asset["size"] <= 0:
                    continue
                url = asset["browser_download_url"]
                digest = asset.get("digest")
                if digest and re.fullmatch(r"sha256:[a-fA-F0-9]{64}", digest):
                    url += "#sha256=" + digest.split(":", 1)[1]
                wheels[channel, name] = {"name": name, "url": url, "version": version}
    return wheels


def write_page(path, title, links):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<!DOCTYPE html>\n<html><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title></head><body>\n"
        f"<h1>{html.escape(title)}</h1>\n"
        + "\n".join(
            f'<a href="{html.escape(url, quote=True)}">{html.escape(label)}</a><br>'
            for url, label in links
        )
        + "\n</body></html>\n",
        encoding="utf-8",
    )


def build_index(release_groups, output):
    wheels = collect_wheels(release_groups)
    if not wheels:
        raise ValueError("No uploaded wheels found; refusing to publish an empty index")
    channels = sorted({channel for channel, name in wheels})
    write_page(output / "index.html", "xllamacpp wheel indexes", [
        (f"whl/{channel}/", channel) for channel in channels
    ])
    write_page(output / "whl" / "index.html", "xllamacpp wheel indexes", [
        (f"{channel}/", channel) for channel in channels
    ])
    for channel in channels:
        directory = output / "whl" / channel
        write_page(directory / "index.html", f"{channel} packages", [("xllamacpp/", "xllamacpp")])
        selected = sorted(
            (asset for (kind, name), asset in wheels.items() if kind == channel),
            key=lambda asset: (asset["version"], asset["name"]), reverse=True,
        )
        write_page(directory / "xllamacpp" / "index.html", f"xllamacpp wheels ({channel})", [
            (asset["url"], asset["name"]) for asset in selected
        ])
        print(f"{channel}: {len(selected)} wheels", flush=True)
    (output / ".nojekyll").touch()
    return wheels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="xorbitsai/xllamacpp")
    parser.add_argument("--fallback", default="xorbitsai/xllamacpp-legacy")
    parser.add_argument("--output", type=Path, default=Path("index"))
    arguments = parser.parse_args()
    groups = []
    if arguments.fallback and arguments.fallback != arguments.source:
        groups.append(list_releases(arguments.fallback, authenticated=False))
    groups.append(list_releases(arguments.source))
    build_index(groups, arguments.output)


if __name__ == "__main__":
    main()
