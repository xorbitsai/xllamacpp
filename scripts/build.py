#!/usr/bin/env python3
"""Build and stage the vendored llama.cpp libraries for xllamacpp."""

from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "thirdparty" / "llama.cpp"
PREFIX = ROOT / "src" / "llama.cpp"
PATCH_DIR = ROOT / "patches" / "llama.cpp"


def log(message: str) -> None:
    print(message, flush=True)


def run(command: list[str], cwd: Path) -> None:
    log("Running: " + subprocess.list2cmdline(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def env_is_set(name: str) -> bool:
    return bool(os.environ.get(name))


def split_cmake_args(value: str) -> list[str]:
    if not value:
        return []
    posix = platform.system() != "Windows"
    try:
        parts = shlex.split(value, posix=posix)
    except ValueError as exc:
        raise SystemExit(f"Invalid CMAKE_ARGS: {exc}") from exc
    if not posix:
        parts = [part.strip("\"'") for part in parts]
    return parts


def llamacpp_patches() -> list[Path]:
    """Local hotfix patches applied to the vendored llama.cpp at build time.

    The submodule checkout itself is never modified permanently: patches from
    patches/llama.cpp/*.patch are applied before building and reverted right
    after, so the tree stays clean for submodule bumps. Once a patch lands
    upstream, delete it (and bump the submodule) -- a patch that no longer
    applies fails the build loudly instead of being silently skipped.
    """
    if not PATCH_DIR.is_dir():
        return []
    return sorted(PATCH_DIR.glob("*.patch"))


def apply_llamacpp_patches(patches: list[Path]) -> list[Path]:
    """Apply patches to the llama.cpp checkout; return the ones applied now.

    Patches that are already present in the working tree (e.g. left over from
    an interrupted build) are skipped and not returned, so they are not
    reverted either.

    If applying a patch fails, every patch applied by this call is reverted
    before the error propagates, so the checkout is never left half-patched.
    """
    applied: list[Path] = []
    try:
        for patch in patches:
            already_applied = (
                subprocess.run(
                    ["git", "apply", "--reverse", "--check", str(patch)],
                    cwd=PROJECT,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).returncode
                == 0
            )
            if already_applied:
                log(f"patch already applied, skipping: {patch.name}")
                continue
            run(["git", "apply", str(patch)], cwd=PROJECT)
            log(f"applied patch: {patch.name}")
            applied.append(patch)
    except Exception:
        log("patch application failed, reverting already-applied patches")
        try:
            revert_llamacpp_patches(applied)
        except Exception as revert_exc:
            log(f"failed to revert partially applied patches: {revert_exc}")
        raise
    return applied


def revert_llamacpp_patches(patches: list[Path]) -> None:
    for patch in reversed(patches):
        run(["git", "apply", "--reverse", str(patch)], cwd=PROJECT)
        log(f"reverted patch: {patch.name}")
        # git apply --reverse restores the original content but with a fresh
        # mtime that is still *older* than the objects just compiled from the
        # patched sources. Bump the mtime of every file the patch touches so
        # the next build recompiles them if the patch set has changed.
        for line in subprocess.run(
            ["git", "apply", "--numstat", str(patch)],
            cwd=PROJECT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines():
            parts = line.split("\t")
            if len(parts) == 3:
                touched = PROJECT / parts[2]
                if touched.exists():
                    os.utime(touched)


def hip_compiler() -> str:
    """Return the path to the HIP C++ compiler (clang).

    On ROCm 7.0+ the hipconfig Perl scripts were removed; if the
    ``hipconfig`` binary is unavailable or fails, fall back to the
    standard ROCm installation path.
    """
    try:
        hip_root = subprocess.check_output(
            ["hipconfig", "-l"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
        hip_root = str(Path(rocm_path) / "llvm")
        log(f"`hipconfig -l` failed ({exc}), falling back to {hip_root}")
    return str(Path(hip_root) / "clang")


def build_llamacpp() -> None:
    log("update from llama.cpp main repo")
    if not PROJECT.exists():
        raise SystemExit(f"Missing llama.cpp checkout: {PROJECT}")

    system = platform.system()
    machine = platform.machine().lower()
    nproc = os.environ.get("NPROC") or str(os.cpu_count() or 2)

    build_dir = PROJECT / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    cmake_args = [
        "-DBUILD_SHARED_LIBS=OFF",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        "-DCMAKE_INSTALL_LIBDIR=lib",
        "-DLLAMA_CURL=OFF",
        "-DLLAMA_LLGUIDANCE=ON",
        "-DLLAMA_BUILD_BORINGSSL=ON",
        "-DLLAMA_OPENSSL=OFF",
    ]
    log("Using BoringSSL (static linking)")

    if env_is_set("XLLAMACPP_RELEASE") and system != "Darwin":
        log("Release mode: disabling native CPU optimizations for portability")
        cmake_args.append("-DGGML_NATIVE=OFF")
    else:
        log("Optimizing for native CPU (GGML_NATIVE=ON by default)")

    if system == "Darwin":
        if not os.environ.get("MACOSX_DEPLOYMENT_TARGET"):
            raise SystemExit("MACOSX_DEPLOYMENT_TARGET must be set for macOS builds")
        cmake_args.append(
            f"-DCMAKE_OSX_DEPLOYMENT_TARGET={os.environ['MACOSX_DEPLOYMENT_TARGET']}"
        )

    user_cmake_args = split_cmake_args(os.environ.get("CMAKE_ARGS", ""))
    cmake_args.extend(user_cmake_args)

    targets = [
        "llama-common-base",
        "llama-common",
        "llama",
        "ggml",
        "ggml-cpu",
        "mtmd",
        "cpp-httplib",
        "server-context",
        "llama-server",
    ]

    if env_is_set("XLLAMACPP_BUILD_CUDA"):
        log("Building for CUDA")
        # CI pipelines pin CUDA_ARCHITECTURES to a curated list to keep build
        # times under the runner limit (a few -real archs + PTX fallbacks).
        #
        # When unset (i.e. a user building locally for their own use), fall back
        # to CMake's "native" keyword. Per the CMake docs this detects the GPUs
        # actually installed on the build machine and compiles SASS only for
        # those architectures. That keeps the build fast and produces fully
        # arch-optimized code for the local hardware -- at the cost of a binary
        # that is not portable to other GPU architectures.
        cuda_archs = os.environ.get("CUDA_ARCHITECTURES") or "native"
        log(f"Using CUDA architectures: {cuda_archs}")
        cmake_args.extend(
            [
                "-DGGML_CUDA=ON",
                "-DGGML_CUDA_FORCE_MMQ=ON",
                f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}",
            ]
        )
        targets.append("ggml-cuda")
    elif env_is_set("XLLAMACPP_BUILD_HIP"):
        log("Building for AMD GPU")
        # CI pipelines pin AMDGPU_TARGETS per ROCm version (see the
        # build-wheel-cuda-hip.yaml matrix). When unset (local builds),
        # fall back to RDNA2 + RDNA3 targets.
        amdgpu_targets = os.environ.get("AMDGPU_TARGETS") or (
            "gfx1100;gfx1101;gfx1102;gfx1030;gfx1031;gfx1032"
        )
        # ROCWMMA flash attention provides significant performance gains
        # on supported archs. CI sets this per version; local builds
        # default to ON. gfx12 support requires ROCm 7.0+ (PR #14202),
        # and the warp mask compile issue is fixed in our submodule
        # (PR #15273).
        rocwmma = os.environ.get("GGML_HIP_ROCWMMA_FATTN") or "ON"
        log(f"Using AMDGPU targets: {amdgpu_targets}")
        log(f"ROCWMMA flash attention: {rocwmma}")
        cmake_args.extend(
            [
                f"-DAMDGPU_TARGETS={amdgpu_targets}",
                f"-DCMAKE_HIP_COMPILER={hip_compiler()}",
                f"-DGGML_HIP_ROCWMMA_FATTN={rocwmma}",
                "-DGGML_HIP=ON",
            ]
        )
        targets.append("ggml-hip")
    elif env_is_set("XLLAMACPP_BUILD_VULKAN"):
        if system == "Darwin":
            cmake_args.append("-DCMAKE_BUILD_RPATH=@loader_path")
            if machine == "x86_64":
                log("Building for Intel with Vulkan")
                cmake_args.extend(["-DGGML_METAL=OFF", "-DGGML_VULKAN=ON"])
                targets.extend(["ggml-blas", "ggml-vulkan"])
            else:
                raise SystemExit(
                    "Building for Apple Silicon with Vulkan is not supported"
                )
        else:
            log("Building with Vulkan")
            cmake_args.append("-DGGML_VULKAN=ON")
            targets.append("ggml-vulkan")
    elif env_is_set("XLLAMACPP_BUILD_AARCH64"):
        log("Building for aarch64")
        cmake_args.append("-DGGML_CPU_ARM_ARCH=armv8-a")
        if "-DGGML_BLAS=ON" in os.environ.get("CMAKE_ARGS", ""):
            log("BLAS is enabled via CMAKE_ARGS, adding ggml-blas to build targets")
            targets.append("ggml-blas")
    elif system == "Darwin":
        cmake_args.append("-DCMAKE_BUILD_RPATH=@loader_path")
        if machine == "x86_64":
            log("Building for Intel")
            cmake_args.append("-DGGML_METAL=OFF")
            targets.append("ggml-blas")
        else:
            log("Building for Apple Silicon")
            cmake_args.append("-DGGML_METAL_EMBED_LIBRARY=ON")
            targets.extend(["ggml-blas", "ggml-metal"])
    else:
        log("Building for non-MacOS CPU")
        if "-DGGML_BLAS=ON" in os.environ.get("CMAKE_ARGS", ""):
            log("BLAS is enabled via CMAKE_ARGS, adding ggml-blas to build targets")
            targets.append("ggml-blas")

    log("Running CMake with arguments: " + " ".join(cmake_args))
    log("Building targets: " + " ".join(targets))

    patches_to_revert: list[Path] = []
    try:
        patches_to_revert = apply_llamacpp_patches(llamacpp_patches())
        run(["cmake", "..", *cmake_args], cwd=build_dir)
        run(
            [
                "cmake",
                "--build",
                ".",
                "--config",
                "Release",
                "--parallel",
                nproc,
                "--target",
                *targets,
            ],
            cwd=build_dir,
        )
    finally:
        revert_llamacpp_patches(patches_to_revert)

    shutil.rmtree(PREFIX, ignore_errors=True)
    run([sys.executable, str(ROOT / "scripts" / "copy_libs.py")], cwd=ROOT)


def main() -> int:
    try:
        build_llamacpp()
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
