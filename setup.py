#!/usr/bin/python3

import os
import platform
import subprocess
from setuptools import Extension, setup

from Cython.Build import cythonize

# -----------------------------------------------------------------------------
# constants

BUILD_CUDA = os.getenv("PYLLAMA_BUILD_CUDA")
NAME = "pyllama-cuda12x" if BUILD_CUDA else "pyllama"
CWD = os.getcwd()

VERSION = "0.0.1"

PLATFORM = platform.system()

LLAMACPP_INCLUDES_DIR = os.path.join(CWD, "src/llama.cpp/include")
LLAMACPP_LIBS_DIR = os.path.join(CWD, "src/llama.cpp/lib")

DEFINE_MACROS = []
EXTRA_COMPILE_ARGS = ["-std=c++14"]
EXTRA_LINK_ARGS = []
EXTRA_OBJECTS = []
INCLUDE_DIRS = [
    "src/pyllama",
    LLAMACPP_INCLUDES_DIR,
    os.path.join(
        CWD, "thirdparty/llama.cpp"
    ),  # For including 'common/base64.hpp' in server/utils.hpp
    os.path.join(CWD, "thirdparty/llama.cpp/examples/server"),
]
LIBRARY_DIRS = [
    LLAMACPP_LIBS_DIR,
]
LIBRARIES = []

if PLATFORM == "Windows":
    LIBRARIES.extend(["common", "llama", "ggml", "ggml-base", "ggml-cpu", "Advapi32"])
    if BUILD_CUDA:
        LIBRARY_DIRS.extend([os.getenv("CUDA_PATH_V12_4", "") + "\\Lib\\x64"])
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
        ]
    )
    if BUILD_CUDA:
        EXTRA_OBJECTS.extend([f"{LLAMACPP_LIBS_DIR}/libggml-cuda.a"])
if PLATFORM == "Darwin":
    EXTRA_OBJECTS.extend(
        [
            f"{LLAMACPP_LIBS_DIR}/libggml-blas.a",
            f"{LLAMACPP_LIBS_DIR}/libggml-metal.a",
        ]
    )

INCLUDE_DIRS.append(os.path.join(CWD, "src/pyllama"))

if PLATFORM == "Darwin":
    # EXTRA_LINK_ARGS.append("-mmacosx-version-min=11")
    # add local rpath
    EXTRA_LINK_ARGS.append("-Wl,-rpath," + LLAMACPP_LIBS_DIR)
    os.environ["LDFLAGS"] = " ".join(
        [
            "-framework Accelerate",
            "-framework Foundation",
            "-framework Metal",
            "-framework MetalKit",
        ]
    )

if PLATFORM == "Linux":
    EXTRA_LINK_ARGS.append("-fopenmp")


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
    "python_requires": ">=3.8",
    # "include_package_data": True,
}


# forces cythonize in this case
subprocess.call("cythonize *.pyx", cwd="src/pyllama", shell=True)

if not os.path.exists("MANIFEST.in"):
    with open("MANIFEST.in", "w") as f:
        f.write("exclude src/pyllama/*.pxd\n")
        f.write("exclude src/pyllama/*.pyx\n")
        f.write("exclude src/pyllama/*.cpp\n")
        f.write("exclude src/pyllama/*.h\n")
        f.write("exclude src/pyllama/py.typed\n")

extensions = [
    mk_extension(
        "pyllama.pyllama", sources=["src/pyllama/pyllama.pyx", "src/pyllama/server.cpp"]
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
