#!/usr/bin/python3

import os
import sys
import platform
import subprocess
from setuptools import Extension, setup

from Cython.Build import cythonize

# -----------------------------------------------------------------------------
# constants

BUILD_CUDA = os.getenv("XLLAMACPP_BUILD_CUDA")
BUILD_HIP = os.getenv("XLLAMACPP_BUILD_HIP")
NAME = "xllamacpp"
# NAME = "xllamacpp-cuda12x" if BUILD_CUDA else "xllamacpp"
CWD = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, CWD)
import versioneer

VERSION = versioneer.get_version()

PLATFORM = platform.system()

LLAMACPP_LIBS_DIR = os.path.join(CWD, "src/llama.cpp/lib")

DEFINE_MACROS = []
if PLATFORM == "Windows":
    EXTRA_COMPILE_ARGS = ["/std:c++17"]
else:
    EXTRA_COMPILE_ARGS = ["-std=c++17"]
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
    os.path.join(CWD, "thirdparty/llama.cpp/tools/server"),
    os.path.join(CWD, "thirdparty/llama.cpp/tools/mtmd"),
    os.path.join(CWD, "thirdparty/llama.cpp/vendor"),
]
LIBRARY_DIRS = [
    LLAMACPP_LIBS_DIR,
]
LIBRARIES = []

if PLATFORM == "Windows":
    LIBRARIES.extend(
        ["common", "llama", "ggml", "ggml-base", "ggml-cpu", "mtmd", "Advapi32"]
    )
    if BUILD_CUDA:
        LIBRARY_DIRS.extend([os.getenv("CUDA_PATH", "") + "\\Lib\\x64"])
        LIBRARIES.extend(["ggml-cuda", "cudart", "cublas", "cublasLt", "cuda"])
else:
    LIBRARIES.extend(["pthread"])
    EXTRA_OBJECTS.extend(
        [
            f"{LLAMACPP_LIBS_DIR}/libcommon.a",
            f"{LLAMACPP_LIBS_DIR}/libllama.a",
            f"{LLAMACPP_LIBS_DIR}/libggml.a",
            f"{LLAMACPP_LIBS_DIR}/libggml-base.a",
            f"{LLAMACPP_LIBS_DIR}/libggml-cpu.a",
            f"{LLAMACPP_LIBS_DIR}/libmtmd.a",
        ]
    )
    if BUILD_CUDA:
        EXTRA_OBJECTS.extend(
            [
                f"{LLAMACPP_LIBS_DIR}/libggml-cuda.a",
            ]
        )
        LIBRARY_DIRS.extend(
            [
                os.getenv("CUDA_PATH", "") + "/lib/stubs",
                os.getenv("CUDA_PATH", "") + "/lib",
            ],
        )
        LIBRARIES.extend(["cudart", "cublas", "cublasLt", "cuda"])
    if BUILD_HIP:
        EXTRA_OBJECTS.extend(
            [
                f"{LLAMACPP_LIBS_DIR}/libggml-hip.a",
            ]
        )
        LIBRARY_DIRS.extend(["/opt/rocm/lib"])
        LIBRARIES.extend(["amdhip64", "hipblas", "rocblas"])

if PLATFORM == "Darwin":
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
        define_macros=define_macros if define_macros else [],
        include_dirs=INCLUDE_DIRS,
        libraries=LIBRARIES,
        library_dirs=LIBRARY_DIRS,
        extra_objects=EXTRA_OBJECTS,
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
        language="c++",
    )


# ----------------------------------------------------------------------------
# COMMON SETUP CONFIG

common = {
    "name": NAME,
    "version": VERSION,
    "description": "A cython wrapper of the llama.cpp inference engine.",
    "python_requires": ">=3.9",
    "cmdclass": versioneer.get_cmdclass(),
    "license": "MIT",
    # "include_package_data": True,
}


# forces cythonize in this case
subprocess.call("cythonize *.pyx", cwd="src/xllamacpp", shell=True)

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
        sources=["src/xllamacpp/xllamacpp.pyx", "src/xllamacpp/server.cpp"],
    ),
]

setup(
    **common,
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "embedsignature": False,  # default: False
            "emit_code_comments": False,  # default: True
            "warn.unused": True,  # default: False
        },
    ),
    package_dir={"": "src"},
)
