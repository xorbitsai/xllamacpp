#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"
llama_dir="${repo_root}/thirdparty/llama.cpp"

channel="latest"
if [[ "${1:-}" == "--channel" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "usage: $0 [--channel latest|stable|nightly]" >&2
    exit 64
  fi
  channel="$2"
  shift 2
fi

if [[ $# -ne 0 || ( "${channel}" != "latest" && "${channel}" != "stable" && "${channel}" != "nightly" ) ]]; then
  echo "usage: $0 [--channel latest|stable|nightly]" >&2
  exit 64
fi

if [[ ! -d "${llama_dir}/.git" && ! -f "${llama_dir}/.git" ]]; then
  echo "error: ${llama_dir} is not a git checkout" >&2
  exit 1
fi

select_latest_tag() {
  local tag_pattern="$1"
  git -C "${llama_dir}" for-each-ref refs/tags \
    --sort=-version:refname \
    --format='%(refname:short)' |
    awk -v pattern="${tag_pattern}" '!found && $0 ~ pattern { print; found = 1 }'
}

case "${channel}" in
  stable)
    latest_tag="$(select_latest_tag '^v[0-9]+\.[0-9]+\.[0-9]+$')"
    tag_description="stable SemVer"
    resolved_channel="stable"
    ;;
  nightly)
    latest_tag="$(select_latest_tag '^b[0-9]+$')"
    tag_description="nightly build"
    resolved_channel="nightly"
    ;;
  latest)
    # The b-number stream is llama.cpp's rolling sequence of current builds.
    # Prefer its highest numeric tag; fall back to stable only if no b tags exist.
    latest_tag="$(select_latest_tag '^b[0-9]+$')"
    if [[ -n "${latest_tag}" ]]; then
      resolved_channel="nightly"
    else
      latest_tag="$(select_latest_tag '^v[0-9]+\.[0-9]+\.[0-9]+$')"
      resolved_channel="stable"
    fi
    tag_description="nightly build or stable SemVer"
    ;;
esac

if [[ -z "${latest_tag}" ]]; then
  echo "error: no llama.cpp ${tag_description} tags found; run update_llamacpp_tags.sh first" >&2
  exit 1
fi

pinned_commit="$(git -C "${llama_dir}" rev-parse HEAD)"
latest_tag_commit="$(git -C "${llama_dir}" rev-parse "${latest_tag}^{}")"

echo "Requested channel:    ${channel}"
echo "Resolved channel:     ${resolved_channel}"
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
