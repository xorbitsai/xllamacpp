#!/usr/bin/python3

import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as setuptools_build_ext

from Cython.Build import cythonize

# -----------------------------------------------------------------------------
# constants

BUILD_CUDA = os.getenv("XLLAMACPP_BUILD_CUDA")
BUILD_HIP = os.getenv("XLLAMACPP_BUILD_HIP")
BUILD_VULKAN = os.getenv("XLLAMACPP_BUILD_VULKAN")
NAME = "xllamacpp"
# NAME = "xllamacpp-cuda12x" if BUILD_CUDA else "xllamacpp"
CWD = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, CWD)
import versioneer

VERSION = versioneer.get_version()

PLATFORM = platform.system()

LLAMACPP_LIBS_DIR = os.path.join(CWD, "src/llama.cpp/lib")
DEFAULT_MACOSX_DEPLOYMENT_TARGET = "13.3"
MACOSX_DEPLOYMENT_TARGET = (
    os.environ.get("MACOSX_DEPLOYMENT_TARGET") or DEFAULT_MACOSX_DEPLOYMENT_TARGET
)
if PLATFORM == "Darwin":
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = MACOSX_DEPLOYMENT_TARGET

# ABI3 (Limited API) support for GIL-enabled Python 3.10+.
# Python 3.14 free-threaded builds do not support the Limited API, so they need
# a dedicated cp314t extension instead.
PY_LIMITED_API_VERSION = 0x030A0000  # Python 3.10
FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))

if FREE_THREADED:
    # Py_GIL_DISABLED is not defined automatically by the Windows headers.
    DEFINE_MACROS = [("Py_GIL_DISABLED", "1")]
else:
    DEFINE_MACROS = [("Py_LIMITED_API", PY_LIMITED_API_VERSION)]
if PLATFORM == "Windows":
    EXTRA_COMPILE_ARGS = ["/std:c++17"]
else:
    EXTRA_COMPILE_ARGS = [
        "-std=c++17",
        "-fvisibility=hidden",
        "-fvisibility-inlines-hidden",
    ]
    if PLATFORM == "Darwin":
        EXTRA_COMPILE_ARGS.append(f"-mmacosx-version-min={MACOSX_DEPLOYMENT_TARGET}")
EXTRA_LINK_ARGS = []
EXTRA_OBJECTS = []
INCLUDE_DIRS = [
    "src/xllamacpp",
    os.path.join(CWD, "thirdparty/llama.cpp/include"),
    os.path.join(CWD, "thirdparty/llama.cpp/common"),
    os.path.join(CWD, "thirdparty/llama.cpp/ggml/include"),
    os.path.join(
        CWD, "thirdparty/llama.cpp"
    ),  # For including 'common/base64.hpp' in server/utils.hpp
    os.path.join(
        CWD, "thirdparty/llama.cpp/build/tools/server"
    ),  # For including index.html.gz.hpp and loading.html.hpp
    os.path.join(
        CWD, "thirdparty/llama.cpp/build/tools/ui"
    ),  # For including generated ui.h
    os.path.join(CWD, "thirdparty/llama.cpp/tools/server"),
    os.path.join(CWD, "thirdparty/llama.cpp/tools/ui"),
    os.path.join(CWD, "thirdparty/llama.cpp/tools/mtmd"),
    os.path.join(CWD, "thirdparty/llama.cpp/vendor"),
]


LIBRARY_DIRS = [
    LLAMACPP_LIBS_DIR,
]
LIBRARIES = []

if PLATFORM == "Windows":
    LIBRARIES.extend(
        [
            "llama-common-base",
            "llama-common",
            "llama",
            "ggml",
            "ggml-base",
            "ggml-cpu",
            "mtmd",
            "vendor-hash",
            "cpp-httplib",
            "server-context",
            "llama-ui",
            "llguidance",
            "ssl",
            "crypto",
            "Advapi32",
            "Shell32",
            "userenv",
            "ntdll",
        ]
    )
    # Note: Windows builds use LLAMA_BUILD_BORINGSSL=ON which statically links OpenSSL
    # Add BoringSSL static libraries for proper symbol resolution
    if BUILD_CUDA:
        LIBRARY_DIRS.extend([os.getenv("CUDA_PATH", "") + "\\Lib\\x64"])
        LIBRARIES.extend(["ggml-cuda", "cudart", "cublas", "cublasLt", "cuda"])
    if BUILD_VULKAN:
        LIBRARY_DIRS.extend([os.getenv("VULKAN_SDK", "") + "\\Lib"])
        LIBRARIES.extend(["ggml-vulkan", "vulkan-1"])
else:
    LIBRARIES.extend(["pthread"])
    # Order matters for static linking: dependents before dependencies, and
    # libssl.a/libcrypto.a must come AFTER libraries that reference OpenSSL
    # symbols (e.g., libcpp-httplib.a, libserver-context.a).
    #
    EXTRA_OBJECTS.extend(
        [
            f"{LLAMACPP_LIBS_DIR}/libserver-context.a",
            f"{LLAMACPP_LIBS_DIR}/libllama-ui.a",
            f"{LLAMACPP_LIBS_DIR}/libcpp-httplib.a",
            f"{LLAMACPP_LIBS_DIR}/libmtmd.a",
            f"{LLAMACPP_LIBS_DIR}/libvendor-hash.a",
            f"{LLAMACPP_LIBS_DIR}/libllama-common-base.a",
            f"{LLAMACPP_LIBS_DIR}/libllama-common.a",
            f"{LLAMACPP_LIBS_DIR}/libllguidance.a",
            f"{LLAMACPP_LIBS_DIR}/libllama.a",
            f"{LLAMACPP_LIBS_DIR}/libggml.a",
            f"{LLAMACPP_LIBS_DIR}/libggml-cpu.a",
            f"{LLAMACPP_LIBS_DIR}/libggml-base.a",
            # BoringSSL static libraries must be last (they are dependencies, not dependents)
            f"{LLAMACPP_LIBS_DIR}/libssl.a",
            f"{LLAMACPP_LIBS_DIR}/libcrypto.a",
        ]
    )
    if BUILD_CUDA:
        EXTRA_OBJECTS.extend(
            [
                f"{LLAMACPP_LIBS_DIR}/libggml-cuda.a",
            ]
        )
        # NVIDIA's Linux packages and the official CUDA images put the runtime
        # in lib64, not lib. CUDA_PATH is also usually unset inside containers,
        # which used to degrade these entries to "/lib/stubs" and "/lib" and
        # make the link fail with "ld: cannot find -lcudart".
        cuda_root = os.getenv("CUDA_PATH") or "/usr/local/cuda"
        for sub in ("lib64", "lib"):
            for cand in (
                os.path.join(cuda_root, sub, "stubs"),
                os.path.join(cuda_root, sub),
            ):
                if os.path.isdir(cand) and cand not in LIBRARY_DIRS:
                    LIBRARY_DIRS.append(cand)
        LIBRARIES.extend(["cudart", "cublas", "cublasLt", "cuda"])
    if BUILD_HIP:
        EXTRA_OBJECTS.extend(
            [
                f"{LLAMACPP_LIBS_DIR}/libggml-hip.a",
            ]
        )
        # ROCm Docker images may install the SDK in a versioned prefix such as
        # /opt/rocm-6.4.1 rather than the incomplete /opt/rocm compatibility
        # prefix. Keep linking consistent with scripts/build.py and CMake.
        rocm_path = os.getenv("ROCM_PATH", "/opt/rocm")
        LIBRARY_DIRS.extend([os.path.join(rocm_path, "lib")])
        LIBRARIES.extend(["amdhip64", "hipblas", "rocblas"])
    if BUILD_VULKAN:
        EXTRA_OBJECTS.extend(
            [
                f"{LLAMACPP_LIBS_DIR}/libggml-vulkan.a",
            ]
        )
        LIBRARIES.extend(["vulkan"])

if PLATFORM == "Darwin":
    EXTRA_LINK_ARGS.append(f"-mmacosx-version-min={MACOSX_DEPLOYMENT_TARGET}")
    EXTRA_LINK_ARGS.append("-Wl,-rpath," + LLAMACPP_LIBS_DIR)
    os.environ["LDFLAGS"] = " ".join(
        [
            "-framework Accelerate",
            "-framework Foundation",
            "-framework Metal",
            "-framework MetalKit",
        ]
    )
    # Both the Intel and ARM platforms need to be linked with BLAS.
    EXTRA_OBJECTS.extend(
        [
            f"{LLAMACPP_LIBS_DIR}/libggml-blas.a",
        ]
    )
    if platform.processor() == "arm":
        EXTRA_OBJECTS.extend(
            [
                f"{LLAMACPP_LIBS_DIR}/libggml-metal.a",
            ]
        )
elif PLATFORM == "Linux":
    # Do not statically link to libstdc++; this will cause compatibility issues.
    # gcc-toolset-14's libgomp.a is not built with -fPIC on both x86_64 and aarch64,
    # so we dynamically link libgomp and exclude it via auditwheel instead.
    EXTRA_LINK_ARGS.extend(["-fopenmp", "-static-libgcc"])
    # Check if BLAS is enabled in environment
    if os.path.exists(f"{LLAMACPP_LIBS_DIR}/libggml-blas.a"):
        print("BLAS is enabled, adding ggml-blas to link targets")
        EXTRA_OBJECTS.extend([f"{LLAMACPP_LIBS_DIR}/libggml-blas.a"])
        EXTRA_LINK_ARGS.extend(["-lopenblas"])

INCLUDE_DIRS.append(os.path.join(CWD, "src/xllamacpp"))


def mk_extension(name, sources, define_macros=None):
    return Extension(
        name=name,
        sources=sources,
        define_macros=DEFINE_MACROS + (define_macros if define_macros else []),
        include_dirs=INCLUDE_DIRS,
        libraries=LIBRARIES,
        library_dirs=LIBRARY_DIRS,
        extra_objects=EXTRA_OBJECTS,
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
        language="c++",
        py_limited_api=not FREE_THREADED,
    )


def _build_llamacpp():
    code = subprocess.call(
        [sys.executable, os.path.join(CWD, "scripts/build.py")], cwd=CWD
    )
    if code:
        raise SystemExit(code)


def _cythonize_extensions(extensions):
    cythonized = cythonize(
        extensions,
        include_path=["src/xllamacpp"],
        compiler_directives={
            "language_level": "3",
            "embedsignature": False,  # default: False
            "emit_code_comments": False,  # default: True
            "warn.unused": True,  # default: False
            "binding": True,  # Required for ABI3+
        },
    )
    for extension in cythonized:
        if not hasattr(extension, "_needs_stub"):
            extension._needs_stub = False
    return cythonized


# ----------------------------------------------------------------------------
# COMMON SETUP CONFIG

cmdclass = versioneer.get_cmdclass()

_build_ext = cmdclass.get("build_ext", setuptools_build_ext)
_sdist = cmdclass["sdist"]


class sdist(_sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        project = Path(base_dir).resolve() / "thirdparty/llama.cpp"
        patches = sorted((Path(CWD) / "patches/llama.cpp").glob("*.patch"))
        git_env = os.environ.copy()
        git_env["GIT_CEILING_DIRECTORIES"] = str(project.parent)

        # setuptools hard-links release files to the checkout; detach files
        # touched by patches so packaging cannot change the vendored checkout.
        targets = set()
        for patch in patches:
            stats = subprocess.check_output(
                ["git", "apply", "--numstat", "-z", str(patch)],
                cwd=project,
                env=git_env,
            )
            targets.update(
                os.fsdecode(entry.split(b"\t", 2)[2])
                for entry in stats.split(b"\0")
                if entry
            )
        for name in targets:
            staged = project / name
            if staged.is_file():
                staged.unlink()
                shutil.copy2(Path(CWD) / "thirdparty/llama.cpp" / name, staged)

        for patch in patches:
            already_applied = subprocess.run(
                ["git", "apply", "--reverse", "--check", str(patch)],
                cwd=project,
                env=git_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode == 0
            if not already_applied:
                subprocess.run(
                    ["git", "apply", str(patch)], cwd=project, env=git_env, check=True
                )


cmdclass["sdist"] = sdist


class build_ext(_build_ext):
    def run(self):
        _build_llamacpp()
        # Archives are staged by build.py, so check optional ones only now.
        for extension in self.distribution.ext_modules:
            optional = {"libllguidance.a", "libssl.a", "libcrypto.a"}
            absent = [
                path
                for path in extension.extra_objects
                if os.path.basename(path) in optional and not os.path.exists(path)
            ]
            if absent:
                print(
                    "xllamacpp: skipping absent optional libraries: "
                    + ", ".join(os.path.basename(path) for path in absent)
                )
                extension.extra_objects = [
                    path for path in extension.extra_objects if path not in absent
                ]

            if BUILD_CUDA and PLATFORM != "Windows":
                # Use CMake's NCCL choice, including libraries under NCCL_ROOT.
                cache = os.path.join(CWD, "thirdparty/llama.cpp/build/CMakeCache.txt")
                with open(cache, encoding="utf-8") as f:
                    entries = {}
                    for line in f:
                        key, sep, value = line.partition("=")
                        if sep and ":" in key:
                            entries[key.split(":", 1)[0]] = value.strip()
                if entries.get("GGML_CUDA_NCCL", "").upper() in {
                    "ON", "TRUE", "YES", "1", "Y"
                }:
                    nccl = entries.get("NCCL_LIBRARY", "")
                    if nccl and not nccl.endswith("-NOTFOUND"):
                        extension.extra_objects.append(nccl)
        self.distribution.ext_modules = _cythonize_extensions(
            self.distribution.ext_modules
        )
        self.extensions = self.distribution.ext_modules
        super().run()


cmdclass["build_ext"] = build_ext

common = {
    "name": NAME,
    "version": VERSION,
    "description": "Python bindings for llama.cpp with native server APIs and prebuilt CPU and GPU wheels",
    "python_requires": ">=3.10",
    "cmdclass": cmdclass,
    "license": "MIT",
    # "include_package_data": True,
}


if not os.path.exists("MANIFEST.in"):
    with open("MANIFEST.in", "w") as f:
        f.write("exclude src/xllamacpp/*.pxd\n")
        f.write("exclude src/xllamacpp/*.pyx\n")
        f.write("exclude src/xllamacpp/*.cpp\n")
        f.write("exclude src/xllamacpp/*.h\n")
        f.write("exclude src/xllamacpp/py.typed\n")

extensions = [
    mk_extension(
        "xllamacpp.xllamacpp",
        sources=[
            "src/xllamacpp/xllamacpp.pyx",
            "src/xllamacpp/server.cpp",
            "thirdparty/llama.cpp/tools/server/server-models.cpp",
            "thirdparty/llama.cpp/tools/server/server-http.cpp",
        ],
    ),
]

bdist_wheel_options = {}
if not FREE_THREADED:
    bdist_wheel_options["py_limited_api"] = "cp310"


setup(
    **common,
    ext_modules=extensions,
    package_dir={"": "src"},
    options={"bdist_wheel": bdist_wheel_options},
)
