---
name: sync-llamacpp-bindings
description: Audit and synchronize xllamacpp's Cython declarations, Python properties, and parameter documentation with the currently vendored llama.cpp headers. Use after a llama.cpp update or when checking for stale fields, changed types, or missing enum members; does not update the vendored checkout.
---

# Sync llama.cpp bindings

Work in the xllamacpp repository, preserving staged and unstaged user changes.
The target is the current `thirdparty/llama.cpp` checkout. Record its tag/commit;
fetching tags, moving the checkout, switching branches, and committing are outside
this skill's scope unless separately requested.

## Audit the current API

Run the bundled auditor from the repository root, using the actual path of this
skill's folder for `<skill-dir>`:

```bash
python3 <skill-dir>/scripts/audit_bindings.py --repo .
```

It uses Clang's JSON AST to inspect current headers, then compiles exact C++ type
checks for every declared field, including otherwise unused Cython declarations.
It also compares named enum membership, checks anonymous exported constants, and
checks direct `self.p` accesses in the Python wrapper classes. It does not edit
project files. `--clang` selects a Clang C++ executable; `--pxd`, `--pyx`, and `--json`
support alternate inputs and a machine-readable report. Exit codes: 0 means the
supported checks passed, 1 means binding drift, 2 means an incomplete audit.
Missing tools, headers, and unsupported syntax must not be reported as a clean
audit. The helper needs initialized vendored sources and a Clang C++ toolchain.

Inspect **all current declarations**, even if the header diff is empty. A removed
member can remain unnoticed in a `.pxd` because Cython does not generate accesses
to unused declarations. An extension build alone cannot detect that case.

Also review header changes relative to the relevant pre-update commit, including
staged changes. For an update still in progress, use:

```bash
git diff HEAD -- src/llama.cpp/include
```

When generated copies are unavailable or stale, compare the corresponding headers
under `thirdparty/llama.cpp` against the previous submodule commit instead. Read
vendored headers as the source of truth. Treat `src/llama.cpp/` as generated,
read-only reference material; regenerate it through the repository's build tools
when necessary. Do not hand-edit either upstream headers or generated copies.

The audit reports native fields not declared on already bound types as **review
notes**, because some omissions are intentional (opaque structs, CLI-only fields,
unsupported callbacks). Review these notes against the header diff: a newly
added field on an already wrapped configuration class normally needs both a
declaration and a Python property. The script cannot infer whether omitted fields
are newly added, or whether a property correctly preserves a field's semantics.

## Synchronize owned files

- Update `src/xllamacpp/xllamacpp.pxd` and `xllamacpp.pyx` together. Match native
  member names, types, signedness, containers, array extents, and enum membership.
- Remove declarations and properties for deleted fields, plus obsolete tests and
  descriptions. Do not retain aliases, computed replacements, or remap a removed
  field onto a similarly named field unless the user requests compatibility.
- Add new members of an already exported enum. Keep deliberately opaque enum
  declarations opaque. Do not expose unrelated native APIs solely to silence
  review notes.
- Check getter/setter conversions, wrapper ownership, and configuration defaults.
  Distinguish type/member changes from initializer-only or enum-value-only
  changes; the latter need review but usually no declaration edit.
- Search owned sources, tests, and docs for removed names and stale descriptions.
  For example, `defrag_thold` was removed upstream, and an old defragmentation
  description had accidentally remained on the unrelated `numa` property.
- The AST/type audit covers fields, enum membership, and direct wrapper accesses,
  not function signatures, callback ABIs, or server behavior. Review changed
  function/callback declarations manually. Server-wrapper synchronization is a
  separate task unless the binding changes require it.

## Verify and finish

Rerun the auditor after edits. If its parser cannot handle a declaration, inspect
the native AST/header and improve the helper or supply an explicit manual check;
do not remove declarations or suppress errors merely to make it pass.
When changing the helper, run `python3 <skill-dir>/scripts/test_audit_bindings.py`;
its integration checks use temporary fixture projects.

After binding changes, compile the actual extension and run the local suite using
the project's configured Python environment:

```bash
python3 setup.py build_ext --inplace
PYTHONPATH=src python3 -m pytest tests -ra
```

Run these with native toolchain and local socket access. The native library build
alone does not validate Cython. A changed source list or stale build artifacts may
require `build_ext --inplace --force`; diagnose the first concrete failure before
patching compatibility code. Add focused regression coverage for observable
behavior changes rather than tests that merely restate declarations.

Review `scripts/generate_params_doc.py` mappings, groups, descriptions, and parser
against the updated bindings. Generate `params_reference.md` with that script;
verify field types/defaults and Python constructor overrides. Regenerate twice
and confirm identical output. Run `tests/test_generate_params_doc.py` if present,
especially after generator edits. Do not edit generated Markdown by hand.

Finish with `git diff --check HEAD` and inspect the final diff without disturbing
the user's index. Report the target tag/commit, audited scope, actual binding and
documentation fixes, intentional omissions, build/test results, and any audit
limitations. Leave changes uncommitted for review unless asked to commit.
