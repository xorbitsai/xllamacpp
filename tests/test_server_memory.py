"""Check OOM memory reports without exhausting real GPU or host memory."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_native_memory_report(tmp_path):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if not compiler:
        pytest.skip("native memory checks require a C++ compiler")
    root = Path(__file__).resolve().parents[1]
    llama = root / "thirdparty/llama.cpp"
    binary = tmp_path / "memory-checks"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            *[
                "-I" + str(path)
                for path in (
                    root / "src/xllamacpp",
                    llama / "common",
                    llama / "include",
                    llama / "ggml/include",
                    llama / "vendor",
                )
            ],
            str(root / "tests/native/server_memory.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, timeout=10)
