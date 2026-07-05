---
name: update-llamacpp
description: Update this xllamacpp repository's vendored llama.cpp checkout. Use when Codex is asked to refresh, update, or sync thirdparty/llama.cpp to the latest upstream llama.cpp release tag whose name starts with b, such as b9704, ensure its nested submodules are initialized and updated, fetch upstream llama.cpp tags, check whether the vendored checkout is already pinned to the latest b* release tag, build xllamacpp after a llama.cpp update, update Cython bindings for changed header fields and enum members, or synchronize generated llama-server changes into the owned Python server wrapper.
---

# Update llama.cpp

## Overview

Use this skill to update the vendored `thirdparty/llama.cpp` checkout in this repository to the newest upstream release tag whose name starts with `b`, for example `b9704`. The checkout update is automated by `scripts/update_xllamacpp.sh`; upstream tag refresh is automated by `scripts/update_llamacpp_tags.sh`; latest `b*` tag comparison is done with the commands in this workflow; xllamacpp build verification is automated by `scripts/build_xllamacpp.sh`; header binding review is assisted by `scripts/check_header_field_bindings.py`. The workflow also audits generated `src/llama.cpp/src/server.cpp` changes against the owned `src/xllamacpp/server.cpp` wrapper.

## Workflow

1. Inspect `git status --short` at the repository root and in `thirdparty/llama.cpp`. Do not discard unrelated user changes.
2. Run the updater from the repository root:

   ```bash
   .codex/skills/update-llamacpp/scripts/update_xllamacpp.sh
   ```

3. The script first switches the parent xllamacpp repository to its upstream default branch from `origin/HEAD` (normally `main`) and fast-forwards it to the latest code. It then switches `thirdparty/llama.cpp` to the upstream default branch from `origin/HEAD` (currently `master` for llama.cpp), fetches it, fast-forwards to it, then runs recursive submodule sync and update inside `thirdparty/llama.cpp`.
4. After it completes, inspect the resulting submodule pointer change from the xllamacpp repository root with `git status --short` and `git diff --submodule=log -- thirdparty/llama.cpp`.
5. Refresh upstream tags when requested:

   ```bash
   .codex/skills/update-llamacpp/scripts/update_llamacpp_tags.sh
   ```

6. The tag script fetches tags from `thirdparty/llama.cpp`'s `origin`, prunes local tags that no longer exist upstream, and force-updates moved upstream tag refs. It does not move branches, commits, or the parent repository's recorded submodule pointer.
7. Check whether the vendored checkout is already pinned to the latest upstream release tag whose name starts with `b`:

   ```bash
   latest_tag="$(
     git -C thirdparty/llama.cpp for-each-ref refs/tags/b* \
       --sort=-creatordate \
       --format='%(refname:short)' \
       --count=1
   )"
   test -n "$latest_tag"
   latest_tag_commit="$(git -C thirdparty/llama.cpp rev-parse "${latest_tag}^{}")"
   pinned_commit="$(git -C thirdparty/llama.cpp rev-parse HEAD)"
   printf 'Latest llama.cpp b* tag: %s\nLatest b* tag commit:   %s\nPinned commit:          %s\n' \
     "$latest_tag" "$latest_tag_commit" "$pinned_commit"
   ```

8. If the latest `b*` tag's peeled commit matches `thirdparty/llama.cpp` `HEAD`, stop: llama.cpp is already at the latest `b*` tagged release. If it differs, pin `thirdparty/llama.cpp` to the exact latest `b*` tag before continuing. The vendored submodule must end on a clean `b*` tag commit, not on upstream branch commits after the tag or on a non-`b*` tag:

   ```bash
   latest_tag="$(
     git -C thirdparty/llama.cpp for-each-ref refs/tags/b* \
       --sort=-creatordate \
       --format='%(refname:short)' \
       --count=1
   )"
   test -n "$latest_tag"
   git -C thirdparty/llama.cpp switch --detach "$latest_tag"
   git -C thirdparty/llama.cpp describe --tags --exact-match HEAD
   ```

9. Build xllamacpp outside the sandbox before editing compatibility code:

   ```bash
   .codex/skills/update-llamacpp/scripts/build_xllamacpp.sh
   ```

10. If the outside-sandbox build fails, inspect the log path printed by the script and use the first concrete CMake or compiler error to drive the next fix. Build failure is expected after many upstream llama.cpp updates. Do not patch build code for failures that only reproduce inside the sandbox.
11. Check changed generated llama.cpp headers for struct/class field additions, removals, C/C++ type changes, and enum member additions/removals. Treat `src/llama.cpp/` as read-only while doing this review:

   ```bash
   .codex/skills/update-llamacpp/scripts/check_header_field_bindings.py
   ```

12. For each reported field or enum change, read the changed header in `src/llama.cpp/` and check `src/xllamacpp/xllamacpp.pxd` plus `src/xllamacpp/xllamacpp.pyx`.
13. If a field was removed, remove every binding for that field from both `src/xllamacpp/xllamacpp.pxd` and `src/xllamacpp/xllamacpp.pyx`. Also remove or update tests that access it. Do not preserve the removed Python name by mapping it to another field, adding a computed property, or creating a compatibility alias unless the user explicitly requests backward compatibility. In particular, do not infer that similarly named fields are semantically interchangeable.
14. If a field changed type, update its Cython declaration and Python property access to match the header exactly.
15. If a new field was added and its owning C++ struct/class already has Python bindings, add the corresponding Cython declaration and Python property binding for the new field.
16. Treat enum members like fields: if a new enum member is added and its enum type already has a Python binding, add the new enum member to the binding; if a removed enum member is bound, remove it.
17. Ignore initializer/default value-only changes and enum value-only changes for now; this step only cares about field type changes, removed fields, added fields, removed enum members, and added enum members.
18. Search `xllamacpp.pxd`, `xllamacpp.pyx`, owned C++ sources, and tests for every removed field name. Confirm no binding implementation or stale test remains. Unrelated uses of a generic name such as `name` may remain.
19. Check whether the generated upstream server changed, including staged changes:

   ```bash
   git diff HEAD -- src/llama.cpp/src/server.cpp
   git diff HEAD -- src/xllamacpp/server.cpp
   ```

20. If `src/llama.cpp/src/server.cpp` changed, review every behavioral hunk and classify it as ported or intentionally inapplicable to the Python wrapper. Do not stop after fixing compiler errors. Audit includes, initialization and teardown, model preparation/download handling, route registration, disabled-endpoint behavior, child/router state handling, streaming lifecycle, and logging.
21. Port applicable server behavior into `src/xllamacpp/server.cpp` in the same relative section and order as upstream. Preserve Python-specific requirements such as avoiding process-wide locale changes, returning initialization status through the promise, clearing callbacks whose captures can outlive the server thread, and keeping router mode disabled when the Python API requires a model path. Add cleanup for every Python-process early return after starting a global upstream subsystem.
22. Add or update focused tests for observable server changes. New routes, response statuses, model preparation paths, and lifecycle behavior require regression coverage. Compare the ordered anchors in both server files after editing; differences must be deliberate and explainable.
23. Compile the Cython extension outside the sandbox after binding or owned C++ edits. The native build wrapper only builds and stages llama.cpp libraries; it does not validate `.pxd`, `.pyx`, or owned C++ compatibility:

   ```bash
   python3 setup.py build_ext --inplace
   ```

   Use the project's configured Python environment if the default `python3` lacks `setuptools` or `Cython`.
24. After the extension build succeeds, run the full local test suite outside the sandbox:

   ```bash
   PYTHONPATH=src python3 -m pytest tests
   ```

25. Fix all test failures caused by the update. Do not treat sandbox-only local socket binding failures or native backend initialization failures as project regressions until the same test also fails outside the sandbox.
26. After the build and tests pass, create the local working branch `enh/update_llama_cpp` from the current repository state. If a local branch with that name already exists, delete it first:

   ```bash
   git branch -D enh/update_llama_cpp
   git switch -c enh/update_llama_cpp
   ```
27. Tell the user the update work is done. Summarize the work, especially the llama.cpp tag and commit, binding changes, server-wrapper synchronization and intentional divergences, and build/test results. Ask the user to review the changes and commit manually.

## Guardrails

- Never modify files under `src/llama.cpp/`. This directory is read-only for the skill; use it only as copied upstream reference material when deciding what to change in xllamacpp-owned files.
- The updater requires no tracked local changes in the parent xllamacpp worktree before it switches branches or pulls. If the parent worktree has tracked changes, stop and ask how to handle them.
- The updater requires a clean `thirdparty/llama.cpp` worktree before it switches branches or pulls. If that checkout is dirty, stop and ask how to handle those local changes.
- Use fast-forward-only pulls. Do not create merge commits while updating the vendored dependency.
- Do not run root-level `git submodule update thirdparty/llama.cpp` after the update, because that would reset the vendored checkout back to the commit recorded by the parent repository.
- Treat tag refresh as remote metadata synchronization only. If local-only tags matter for a task, inspect them before running the tag script because `--prune-tags` removes tags absent from `origin`.
- Run `scripts/update_llamacpp_tags.sh` before selecting the latest `b*` tag when current upstream tag state matters.
- Pin `thirdparty/llama.cpp` to the exact latest `b*` tag before compatibility work. Do not leave it on `master`, another branch commit beyond the tag, or a non-`b*` tag.
- On macOS, the build wrapper defaults `MACOSX_DEPLOYMENT_TARGET` to `13.3` only when it is unset, matching this repository's wheel workflow. Caller-provided build environment variables still take precedence.
- Run the xllamacpp build outside the sandbox. llama.cpp's CMake build may provision UI assets or initialize native build tooling differently under sandbox restrictions.
- Treat removed upstream fields as removed public bindings by default. Never silently retain them as aliases or computed properties; backward compatibility requires an explicit user request.
- Treat `src/llama.cpp/src/server.cpp` as generated read-only reference code. Implement server compatibility only in `src/xllamacpp/server.cpp`.
- Never assume a successful compile means the server wrapper is synchronized. Review every generated server hunk and keep applicable blocks in upstream-relative order.
- Do not copy CLI-only behavior blindly. Preserve Python process lifetime, error propagation, and the explicit router-mode policy, and document each intentional omission.
- Run the in-place Cython extension build outside the sandbox after changing bindings or owned C++ compatibility code.
- Run the full `tests/` suite outside the sandbox. The server tests bind local HTTP sockets and native backend initialization can behave differently under sandbox restrictions.
- Only delete the local `enh/update_llama_cpp` branch when recreating the update branch. Do not delete any remote branch unless the user explicitly asks.
- Do not create a commit as part of this skill unless the user explicitly asks. End with a summary of the work, including the updated llama.cpp tag/commit, then prompt the user to review the changes and commit manually.
- Treat `scripts/check_header_field_bindings.py` as a review aid, not an automatic patcher. Verify each reported field or enum member against the header and bindings before editing Cython code.
