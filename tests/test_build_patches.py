"""Unit tests for the build-time llama.cpp patch machinery in scripts/build.py.

The wheel build applies hotfix patches from patches/llama.cpp/*.patch to the
vendored submodule before the CMake build and reverts them right after, so
the submodule working tree stays pristine. These tests exercise that logic
against a throwaway git repository instead of the real submodule.
"""

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

spec = importlib.util.spec_from_file_location(
    "xllamacpp_scripts_build", ROOT / "scripts" / "build.py"
)
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def run_git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )


@pytest.fixture
def fake_llamacpp(tmp_path, monkeypatch):
    """A throwaway git repo standing in for the llama.cpp submodule.

    The patch file is authored the same way as the real one: edit, git diff,
    revert.
    """
    proj = tmp_path / "llama.cpp"
    proj.mkdir()
    run_git(proj, "init")
    target = proj / "ggml-metal-device.m"
    target.write_text("line one\nline two\n")
    run_git(proj, "add", ".")
    run_git(
        proj,
        "-c",
        "user.email=test@example.com",
        "-c",
        "user.name=test",
        "commit",
        "-m",
        "init",
    )

    target.write_text("line one\nline two patched\n")
    patch_dir = tmp_path / "patches" / "llama.cpp"
    patch_dir.mkdir(parents=True)
    patch_file = patch_dir / "0001-test.patch"
    patch_file.write_text(run_git(proj, "diff").stdout)
    run_git(proj, "checkout", "--", ".")

    monkeypatch.setattr(build, "PROJECT", proj)
    monkeypatch.setattr(build, "PATCH_DIR", patch_dir)
    return proj, target, patch_file


def git_is_clean(proj: Path) -> bool:
    return run_git(proj, "status", "--porcelain").stdout == ""


def test_no_patch_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "PATCH_DIR", tmp_path / "does-not-exist")
    assert build.llamacpp_patches() == []


def test_apply_then_revert_leaves_tree_clean(fake_llamacpp):
    proj, target, patch_file = fake_llamacpp

    patches = build.llamacpp_patches()
    assert patches == [patch_file]

    applied = build.apply_llamacpp_patches(patches)
    assert applied == [patch_file]
    assert target.read_text() == "line one\nline two patched\n"

    build.revert_llamacpp_patches(applied)
    assert target.read_text() == "line one\nline two\n"
    assert git_is_clean(proj)


def test_apply_is_idempotent(fake_llamacpp):
    proj, target, patch_file = fake_llamacpp

    applied_first = build.apply_llamacpp_patches(build.llamacpp_patches())
    assert applied_first == [patch_file]

    # second build against an already-patched tree: skipped, not reverted
    applied_second = build.apply_llamacpp_patches(build.llamacpp_patches())
    assert applied_second == []
    assert target.read_text() == "line one\nline two patched\n"

    build.revert_llamacpp_patches(applied_first)
    assert git_is_clean(proj)


def test_revert_only_what_was_applied(fake_llamacpp):
    proj, target, patch_file = fake_llamacpp

    build.apply_llamacpp_patches(build.llamacpp_patches())
    # simulate an interrupted build: patch left applied, next build skips it
    applied = build.apply_llamacpp_patches(build.llamacpp_patches())
    build.revert_llamacpp_patches(applied)
    # nothing was applied by this run, so nothing is reverted either
    assert target.read_text() == "line one\nline two patched\n"


def test_inapplicable_patch_fails_loudly(fake_llamacpp, tmp_path):
    proj, target, patch_file = fake_llamacpp
    patch_file.write_text(
        "diff --git a/ggml-metal-device.m b/ggml-metal-device.m\n"
        "--- a/ggml-metal-device.m\n"
        "+++ b/ggml-metal-device.m\n"
        "@@ -1,2 +1,2 @@\n"
        "-this content does not exist\n"
        "-neither does this\n"
        "+garbage\n"
        "+patch\n"
    )
    with pytest.raises(subprocess.CalledProcessError):
        build.apply_llamacpp_patches(build.llamacpp_patches())
