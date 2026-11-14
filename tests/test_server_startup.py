"""Exercise the native retry controller without exhausting real GPU memory."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import xllamacpp as xlc


@pytest.mark.parametrize(
    "margins,error",
    [
        ([], ValueError),
        ((), ValueError),
        ([0], ValueError),
        ([-1], ValueError),
        ([2, 2], ValueError),
        ([2, 1], ValueError),
        ([True], TypeError),
        ([1.5], TypeError),
        (["256 MiB"], TypeError),
        ([2 * sys.maxsize + 2], ValueError),
        (True, TypeError),
        (False, TypeError),
        (3, TypeError),
        ({}, TypeError),
        ("1024", TypeError),
        ("automatic", TypeError),
        (iter([1, 2]), TypeError),
    ],
)
def test_server_rejects_invalid_retry_margins(margins, error):
    with pytest.raises(error):
        xlc.Server(xlc.CommonParams(), fit_retry_margins_bytes=margins)


@pytest.fixture(scope="module")
def startup_checks(tmp_path_factory):
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if not compiler:
        pytest.skip("native retry checks require a C++ compiler")
    root = Path(__file__).resolve().parents[1]
    binary = tmp_path_factory.mktemp("startup") / "startup-checks"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(root / "src/xllamacpp"),
            str(root / "tests/native/server_startup.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


@pytest.mark.parametrize(
    "scenario",
    ["classification", "retry", "adaptive", "adaptive_failures", "failures", "margin"],
)
def test_native_startup_policy(startup_checks, scenario):
    subprocess.run([str(startup_checks), scenario], check=True, timeout=10)


@pytest.mark.parametrize(
    "margins", ["default", None, "auto", (256 << 20, 512 << 20), [256 << 20, 512 << 20]]
)
def test_invalid_model_diagnostics(tmp_path, margins):
    params = xlc.CommonParams()
    # A filename that mentions OOM must not trigger allocation classification.
    params.model.path = str(tmp_path / "out of memory.gguf")
    params.port = 0
    params.n_ctx = 128
    params.n_parallel = 1
    params.warmup = False
    params.verbosity = 0
    with pytest.raises(RuntimeError) as error:
        if margins == "default":
            xlc.Server(params)
        else:
            xlc.Server(params, fit_retry_margins_bytes=margins)
    message = str(error.value)
    assert "after 1 attempt(s)" in message
    assert "model loading" in message
    assert "out of memory.gguf" in message
    assert "failed to load model" in message


@pytest.mark.parametrize("container", [tuple, list, None])
def test_retry_preserves_explicit_settings(model_path, container):
    params = xlc.CommonParams()
    params.model.path = str(Path(model_path) / "stories15M-q4_0.gguf")
    params.port = 0
    params.n_ctx = 128
    params.n_parallel = 1
    params.n_gpu_layers = 0
    params.tensor_split = [1.0] + [0.0] * 127
    params.warmup = False
    before = (
        params.n_ctx,
        params.n_gpu_layers,
        params.tensor_split,
        params.fit_params_target,
    )
    margins = container((256 << 20, 512 << 20)) if container else "auto"
    server = xlc.Server(params, fit_retry_margins_bytes=margins)
    assert server.startup_attempts == 1
    assert server.listening_address.startswith("http://127.0.0.1:")
    assert (
        params.n_ctx,
        params.n_gpu_layers,
        params.tensor_split,
        params.fit_params_target,
    ) == before
    assert margins == (container((256 << 20, 512 << 20)) if container else "auto")


@pytest.mark.parametrize("mode", ["default", "auto", None])
def test_auto_retry_first_load_success(model_path, mode):
    params = xlc.CommonParams()
    params.model.path = str(Path(model_path) / "stories15M-q4_0.gguf")
    params.port = 0
    params.n_ctx = 128
    params.n_parallel = 1
    params.warmup = False
    params.verbosity = 0
    margins = params.fit_params_target
    if mode == "default":
        server = xlc.Server(params)
    else:
        server = xlc.Server(params, fit_retry_margins_bytes=mode)
    assert server.startup_attempts == 1
    assert server.listening_address.startswith("http://127.0.0.1:")
    assert params.fit_params_target == margins
    assert params.n_gpu_layers == -1
