"""Exercise the patched llguidance ExternalProject without network access."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(command, cwd, env=None):
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


@pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("cargo", "cmake", "git")),
    reason="requires Cargo, CMake and Git",
)
@pytest.mark.parametrize("redirect_target_dir", [False, True])
def test_llguidance_stages_header_and_library(tmp_path, redirect_target_dir):
    # Cargo 1.100 no longer leaves the header in target/release. This fixture
    # leaves it only in parser/ so CMake must stage it independently of Cargo.
    upstream = tmp_path / "upstream"
    (upstream / "parser").mkdir(parents=True)
    (upstream / "parser" / "llguidance.h").write_text("int llguidance_test(void);\n")
    (upstream / "Cargo.toml").write_text(
        '[package]\nname = "llguidance"\nversion = "0.0.0"\n'
        '[lib]\npath = "lib.rs"\ncrate-type = ["staticlib"]\n'
    )
    (upstream / "lib.rs").write_text(
        '#[no_mangle]\npub extern "C" fn llguidance_test() -> i32 { 42 }\n'
    )
    run(["git", "init"], upstream)
    run(["git", "add", "."], upstream)
    run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "fixture",
        ],
        upstream,
    )
    revision = run(["git", "rev-parse", "HEAD"], upstream)

    vendor = tmp_path / "vendor"
    (vendor / "common").mkdir(parents=True)
    shutil.copy2(
        ROOT / "thirdparty/llama.cpp/common/CMakeLists.txt",
        vendor / "common/CMakeLists.txt",
    )
    run(["git", "init"], vendor)
    run(
        [
            "git",
            "apply",
            str(ROOT / "patches/llama.cpp/0003-llguidance-stage-header.patch"),
        ],
        vendor,
    )
    common = (vendor / "common/CMakeLists.txt").read_text()
    block = "if (LLAMA_LLGUIDANCE)" + common.split("if (LLAMA_LLGUIDANCE)", 1)[1]
    block = block.split("\ntarget_link_libraries(${TARGET} PUBLIC llama", 1)[0]
    block = block.replace(
        "https://github.com/guidance-ai/llguidance", upstream.as_posix()
    )
    block = block.replace("d795912fedc7d393de740177ea9ea761e7905774", revision)
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\nproject(llguidance_test CXX)\n"
        "set(TARGET consumer)\nadd_executable(consumer main.cpp)\n"
        "set(LLAMA_LLGUIDANCE ON)\n" + block
    )
    (tmp_path / "main.cpp").write_text(
        'extern "C" {\n#include "llguidance.h"\n}\n'
        "int main() { return llguidance_test() == 42 ? 0 : 1; }\n"
    )
    env = os.environ.copy()
    env["CARGO_NET_OFFLINE"] = "true"
    if redirect_target_dir:
        env["CARGO_TARGET_DIR"] = str(tmp_path / "redirected-target")

    run(["cmake", "-S", ".", "-B", "build"], tmp_path, env)
    run(["cmake", "--build", "build", "--config", "Release"], tmp_path, env)
    output = tmp_path / "build/llguidance/source/target/release"
    assert (output / "llguidance.h").read_bytes() == (
        upstream / "parser/llguidance.h"
    ).read_bytes()
    assert (
        output / ("llguidance.lib" if os.name == "nt" else "libllguidance.a")
    ).is_file()
    executable = "build/Release/consumer.exe" if os.name == "nt" else "build/consumer"
    run([str(tmp_path / executable)], tmp_path)
