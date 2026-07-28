"""Regression tests for Metal bf16 kernel support on Apple Silicon (M3+).

Background: the wheel embeds the Metal shader source and JIT-compiles it at
runtime. llama.cpp injects GGML_METAL_HAS_BF16 when the GPU supports bfloat,
but never pins MTLCompileOptions.languageVersion, so the Metal compiler
derives the default shading language version from the *host executable's*
LC_BUILD_VERSION (the python interpreter). Interpreters linked against old
SDKs (conda, python.org) default to MSL 2.x, in which case the
`__METAL_VERSION__ < 310` guard in ggml-metal.metal silently strips all bf16
kernels and loading a model with BF16 tensors fails with:

    Function kernel_mul_mv_ext_bf16_f32_r1_2 was not found in the library

The fix (patches/llama.cpp/0001-metal-pin-msl-language-version.patch, applied
at build time by scripts/build.py) pins the MSL version to >= 3.1 when the
device reports bfloat support, and probes bf16 compilation at device init: if
the environment cannot compile MSL 3.1, bfloat is disabled (with a warning) so
BF16 ops fall back to CPU instead of failing model loads.

These tests build a tiny llama GGUF whose weights are all BF16 and load it:
1. in-process, and
2. under a python host whose LC_BUILD_VERSION is rewritten to an old SDK,
   emulating conda/python.org interpreters (the failing scenario).

On machines without bfloat support (M1/M2) llama.cpp falls back to CPU for
BF16 ops, so the tests still pass but do not exercise the Metal bf16 kernels.
On M3+ machines the tests additionally verify that the bf16 kernels were
actually built: a silent fallback (bfloat disabled, model running on CPU)
must fail the test, not pass it.
"""

import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Metal bf16 kernels are only exercised on Apple Silicon",
)


def _apple_silicon_gen() -> int:
    """SoC generation (3 for M3, 4 for M4, ...); 0 if unknown."""
    try:
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0
    m = re.search(r"Apple M(\d+)", brand)
    return int(m.group(1)) if m else 0


# bfloat (Metal3 GPU family) requires M3 or newer
_EXPECT_BF16 = _apple_silicon_gen() >= 3


def _f32_to_bf16(arr):
    """Round-to-nearest-even float32 -> bfloat16 bits as uint16."""
    import numpy as np

    bits = arr.astype(np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def _write_bf16_variant(src: Path, dst: Path) -> None:
    """Copy a llama GGUF with every tensor rewritten as BF16 random weights."""
    gguf = pytest.importorskip("gguf")
    import numpy as np

    reader = gguf.GGUFReader(str(src))
    writer = gguf.GGUFWriter(str(dst), arch="llama")
    for field in reader.fields.values():
        # GGUF.* and general.architecture are managed by the writer; file_type
        # describes the original quantization and no longer applies.
        if field.name.startswith("GGUF.") or field.name in (
            "general.architecture",
            "general.file_type",
            "general.quantization_version",
        ):
            continue
        vtype = field.types[0]
        if vtype == gguf.GGUFValueType.ARRAY:
            writer.add_array(field.name, field.contents())
        elif vtype == gguf.GGUFValueType.STRING:
            writer.add_string(field.name, field.contents())
        else:
            writer.add_key_value(field.name, field.contents(), vtype)

    rng = np.random.default_rng(0)
    for tensor in reader.tensors:
        # reader shapes are in ggml order (ne0 first); the writer expects
        # numpy order and reverses them when writing the file.
        shape = tuple(int(d) for d in tensor.shape[::-1])
        data = rng.normal(0.0, 0.02, size=shape).astype(np.float32)
        writer.add_tensor(
            tensor.name,
            _f32_to_bf16(data).reshape(shape),
            raw_dtype=gguf.GGMLQuantizationType.BF16,
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()


@pytest.fixture(scope="session")
def bf16_model_path(tmp_path_factory):
    src = ROOT / "models" / "stories15M-q4_0.gguf"
    if not src.exists():
        pytest.skip(f"reference model not found: {src} (run `make download`)")
    dst = tmp_path_factory.mktemp("bf16-model") / "tiny-llama-bf16.gguf"
    _write_bf16_variant(src, dst)
    return str(dst)


def _load_model(model_path: str) -> None:
    import xllamacpp as xlc

    params = xlc.CommonParams()
    params.model.path = model_path
    params.n_ctx = 512
    params.n_gpu_layers = 99
    # default warmup=True: the warmup run is what compiles the bf16 pipelines
    server = xlc.Server(params)
    del server


def test_metal_bf16_model_loads(bf16_model_path, capfd):
    _load_model(bf16_model_path)
    if _EXPECT_BF16:
        # a silent fallback (bfloat disabled -> BF16 ops on CPU) would still
        # load successfully - make sure that did not happen
        assert "disabling bfloat support" not in capfd.readouterr().err


_PYHOST_C = "#include <Python.h>\nint main(int argc, char **argv) { return Py_BytesMain(argc, argv); }\n"


@pytest.fixture(scope="session")
def old_sdk_python(tmp_path_factory):
    """A python executable with an old LC_BUILD_VERSION (like conda/python.org).

    Metal derives the default shading language version from the main
    executable's LC_BUILD_VERSION; rewriting it to SDK 11.0 reproduces the
    environment in which the bf16 kernels were silently dropped pre-fix.
    """
    for tool in ("clang", "vtool", "codesign"):
        if not shutil.which(tool):
            pytest.skip(f"{tool} is required to build the old-SDK python host")

    tmp = tmp_path_factory.mktemp("pyhost")
    src = tmp / "pyhost.c"
    src.write_text(_PYHOST_C)

    include = sysconfig.get_paths()["include"]
    libdir = sysconfig.get_config_var("LIBDIR")
    match = re.fullmatch(
        r"lib(.+?)\.(?:dylib|so(?:\..*)?|a)", sysconfig.get_config_var("LDLIBRARY") or ""
    )
    lib = match.group(1) if match else f"python{sys.version_info.major}.{sys.version_info.minor}"

    modern = tmp / "pyhost_modern"
    subprocess.run(
        [
            "clang",
            "-mmacosx-version-min=11.0",
            str(src),
            "-o",
            str(modern),
            f"-I{include}",
            f"-L{libdir}",
            f"-l{lib}",
            f"-Wl,-rpath,{libdir}",
        ],
        check=True,
    )

    old = tmp / "python_old"
    subprocess.run(
        ["vtool", "-set-build-version", "1", "11.0", "11.0", "-output", str(old), str(modern)],
        check=True,
    )
    subprocess.run(
        ["codesign", "--force", "--sign", "-", str(old)],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(old), "--version"], check=True, capture_output=True)
    return str(old)


def test_metal_bf16_model_loads_with_old_sdk_python(old_sdk_python, bf16_model_path):
    code = (
        "import xllamacpp as xlc\n"
        "p = xlc.CommonParams()\n"
        f"p.model.path = {bf16_model_path!r}\n"
        "p.n_ctx = 512\n"
        "p.n_gpu_layers = 99\n"
        "xlc.Server(p)\n"
        "print('MODEL_LOADED_OK')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run(
        [old_sdk_python, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert proc.returncode == 0, f"Process failed with code {proc.returncode}. Stderr: {proc.stderr}"
    assert "was not found in the library" not in proc.stderr
    assert "MODEL_LOADED_OK" in proc.stdout
    if _EXPECT_BF16:
        # the bf16 kernels must have been built for real: bfloat must have
        # stayed enabled - a silent disable would still load OK via CPU
        # fallback (with bfloat enabled but no bf16 kernels in the library,
        # the load above would have failed with "was not found")
        assert "disabling bfloat support" not in proc.stderr
