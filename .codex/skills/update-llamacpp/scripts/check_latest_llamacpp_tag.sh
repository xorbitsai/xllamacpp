#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"
llama_dir="${repo_root}/thirdparty/llama.cpp"

channel="stable"
if [[ "${1:-}" == "--channel" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "usage: $0 [--channel stable|nightly]" >&2
    exit 64
  fi
  channel="$2"
  shift 2
fi

if [[ $# -ne 0 || ( "${channel}" != "stable" && "${channel}" != "nightly" ) ]]; then
  echo "usage: $0 [--channel stable|nightly]" >&2
  exit 64
fi

if [[ ! -d "${llama_dir}/.git" && ! -f "${llama_dir}/.git" ]]; then
  echo "error: ${llama_dir} is not a git checkout" >&2
  exit 1
fi

if [[ "${channel}" == "stable" ]]; then
  tag_pattern='^v[0-9]+\.[0-9]+\.[0-9]+$'
  tag_description="stable SemVer"
else
  tag_pattern='^b[0-9]+$'
  tag_description="nightly build"
fi

latest_tag="$(
  git -C "${llama_dir}" for-each-ref refs/tags \
    --sort=-version:refname \
    --format='%(refname:short)' |
    awk -v pattern="${tag_pattern}" '!found && $0 ~ pattern { print; found = 1 }'
)"

if [[ -z "${latest_tag}" ]]; then
  echo "error: no llama.cpp ${tag_description} tags found; run update_llamacpp_tags.sh first" >&2
  exit 1
fi

pinned_commit="$(git -C "${llama_dir}" rev-parse HEAD)"
latest_tag_commit="$(git -C "${llama_dir}" rev-parse "${latest_tag}^{}")"

echo "Release channel:      ${channel}"
echo "Latest llama.cpp tag: ${latest_tag}"
echo "Latest tag commit:   ${latest_tag_commit}"
echo "Pinned commit:       ${pinned_commit}"

counterpart_tags="$(
  git -C "${llama_dir}" tag --points-at "${latest_tag_commit}" |
    awk -v selected="${latest_tag}" '
      $0 != selected && ($0 ~ /^b[0-9]+$/ || $0 ~ /^v[0-9]+\.[0-9]+\.[0-9]+$/) { print }
    '
)"
if [[ -n "${counterpart_tags}" ]]; then
  echo "Equivalent tag(s):"
  while IFS= read -r counterpart_tag; do
    echo "  ${counterpart_tag}"
  done <<< "${counterpart_tags}"
fi
echo

if [[ "${latest_tag_commit}" == "${pinned_commit}" ]]; then
  echo "llama.cpp is already pinned to the latest tag."
  exit 0
fi

echo "llama.cpp is not pinned to the latest tag."
exit 2
