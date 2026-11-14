#!/usr/bin/env python3
"""Compare the production estimator and Server with independent Metal accounting.

Run after `python setup.py build_ext --inplace`. Each case runs in a fresh
process. The optional allocation-limit cases inject allocation failures without
exhausting system memory; they are not physical OOM measurements.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MIB = 1 << 20
GIB = 1 << 30


def build_harness(binary, log):
    llama = ROOT / "thirdparty/llama.cpp"
    build_dirs = sorted(
        (ROOT / "build").glob("temp.*"), key=lambda path: path.stat().st_mtime
    )
    objects = [
        "server-connect.o",
        "server-http.o",
        "server-models.o",
        "server-models-sched.o",
    ]
    object_dir = next(
        (
            directory / "thirdparty/llama.cpp/tools/server"
            for directory in reversed(build_dirs)
            if all(
                (directory / "thirdparty/llama.cpp/tools/server" / name).exists()
                for name in objects
            )
        ),
        None,
    )
    if object_dir is None:
        raise RuntimeError(
            "Build the extension first: python setup.py build_ext --inplace"
        )
    command = ["c++", "-std=c++17", "-O2", "-fno-objc-arc"]
    cache = (llama / "build/CMakeCache.txt").read_text()
    if "LLAMA_SUBPROCESS:BOOL=ON" in cache:
        command.append("-DLLAMA_SUBPROCESS=1")
    includes = [
        ROOT / "src/xllamacpp",
        llama,
        llama / "common",
        llama / "include",
        llama / "ggml/include",
        llama / "vendor",
        llama / "tools/server",
        llama / "tools/mtmd",
        llama / "tools/ui",
        llama / "build/tools/server",
        llama / "build/tools/ui",
    ]
    command += ["-I" + str(path) for path in includes]
    command += [
        str(ROOT / "tests/native/server_memory_accuracy.mm"),
        str(ROOT / "src/xllamacpp/server.cpp"),
        *[str(object_dir / name) for name in objects],
    ]
    libraries = [
        "server-context",
        "llama-ui",
        "cpp-httplib",
        "mtmd",
        "vendor-hash",
        "llama-common-base",
        "llama-common",
        "llguidance",
        "llama",
        "ggml",
        "ggml-cpu",
        "ggml-base",
        "ssl",
        "crypto",
        "ggml-blas",
        "ggml-metal",
    ]
    command += [
        str(ROOT / "src/llama.cpp/lib" / f"lib{name}.a")
        for name in libraries
        if (ROOT / "src/llama.cpp/lib" / f"lib{name}.a").exists()
    ]
    for framework in ["Accelerate", "Foundation", "Metal", "MetalKit", "Security"]:
        command += ["-framework", framework]
    command += ["-lpthread", "-o", str(binary)]
    with log.open("w") as output:
        subprocess.run(command, stdout=output, stderr=output, check=True, cwd=ROOT)


def cases():
    stories = "stories15M-q4_0.gguf"
    qwen = "qwen1_5-0_5b-chat-q2_k.gguf"
    llama = "Llama-3.2-1B-Instruct-Q8_0.gguf"
    entries = [
        ("stories-512", stories, {"n_ctx": 512}),
        ("stories-4096", stories, {"n_ctx": 4096}),
        ("stories-partial", stories, {"n_ctx": 2048, "n_gpu_layers": 3}),
        ("stories-cpu", stories, {"n_ctx": 2048, "n_gpu_layers": 0}),
        ("stories-parallel4", stories, {"n_ctx": 4096, "n_parallel": 4}),
        ("stories-no-flash", stories, {"n_ctx": 2048, "flash_attn": 0, "n_batch": 512}),
        (
            "stories-q8-kv",
            stories,
            {"n_ctx": 4096, "quantized_kv": True, "flash_attn": 1},
        ),
        ("qwen-1024", qwen, {"n_ctx": 1024}),
        ("qwen-8192", qwen, {"n_ctx": 8192}),
        ("qwen-partial", qwen, {"n_ctx": 2048, "n_gpu_layers": 8}),
        ("llama-512", llama, {"n_ctx": 512}),
        ("llama-4096", llama, {"n_ctx": 4096}),
        ("llama-partial", llama, {"n_ctx": 2048, "n_gpu_layers": 8}),
        ("llama-q8-kv", llama, {"n_ctx": 4096, "quantized_kv": True, "flash_attn": 1}),
        ("moe-2048", "stories15M_MOE-F16.gguf", {"n_ctx": 2048}),
    ]
    return [
        {
            "name": name,
            "model": str(ROOT / "models" / model),
            **({"prompt_repeats": 8} if model.startswith("stories") else {}),
            **settings,
        }
        for name, model, settings in entries
    ]


def run_case(binary, config, directory):
    name = config["name"]
    config_path = directory / f"{name}.config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    process = subprocess.run(
        [str(binary), str(config_path)], capture_output=True, text=True, timeout=180
    )
    (directory / f"{name}.log").write_text(process.stderr + process.stdout)
    prefix = "XLLAMACPP_MEMORY_BENCHMARK="
    for line in reversed(process.stdout.splitlines()):
        if line.startswith(prefix):
            result = json.loads(line[len(prefix) :])
            result["exit_code"] = process.returncode
            if process.returncode:
                result["success"] = False
                result["error"] = (
                    f"Process exited {process.returncode} after reporting results"
                )
            print(
                f"{name}: success={result['success']}, "
                f"estimate={mib(result['predicted_device_bytes'])} MiB, "
                f"peak={mib(result['peak_device_bytes'])} MiB, "
                f"attempts={result.get('startup_attempts', 'failed')}",
                flush=True,
            )
            return result
    progress = next(
        (
            json.loads(line.split("=", 1)[1])
            for line in process.stdout.splitlines()
            if line.startswith("XLLAMACPP_MEMORY_ESTIMATE=")
        ),
        {},
    )
    result = {
        **progress,
        "config": config,
        "success": False,
        "exit_code": process.returncode,
        "crashed": process.returncode < 0,
        "peak_device_bytes": None,
        "peak_device_total_bytes": None,
        "error": f"Process exited {process.returncode}; see {name}.log",
    }
    result.setdefault("predicted_device_bytes", None)
    print(f"{name}: process exited {process.returncode}", flush=True)
    return result


def mib(value):
    return f"{value / MIB:.2f}" if value is not None else "n/a"


def write_report(results, output):
    rows = [
        "# Metal memory estimation measurements",
        "",
        "The production estimator is compared with `MTLDevice.currentAllocatedSize` in fresh processes.",
        "Buffer creation is instrumented to capture allocation peaks; a 1 ms sampler also runs through startup and inference.",
        "Each successful case runs a real Server, prompt evaluation, and up to 16 generated tokens.",
        "",
        "The peak is a Metal allocation high-water mark, not physical system RAM usage. CPU buffers are excluded from both sides.",
        "Incremental peaks subtract the process's pre-estimation Metal allocation baseline. Total peaks include it, so backend initialization overhead remains visible.",
        "",
        "| Configuration | Estimated MiB | Incremental peak MiB | Total peak MiB | Error vs total peak | Inference |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        if "allocation_limit_bytes" in result["config"]:
            continue
        predicted = result["predicted_device_bytes"]
        actual = result["peak_device_bytes"]
        if predicted is None or actual is None:
            rows.append(
                f"| {result['config']['name']} | n/a | n/a | n/a | n/a | estimation failed; load {'passed' if result['success'] else 'failed'} |"
            )
            continue
        total_peak = result["peak_device_total_bytes"]
        error = (
            f"{(predicted - total_peak) / total_peak * 100:+.2f}%"
            if predicted and total_peak
            else "n/a"
        )
        rows.append(
            f"| {result['config']['name']} | {predicted / MIB:.2f} | "
            f"{actual / MIB:.2f} | {total_peak / MIB:.2f} | "
            f"{error} | {'pass' if result['success'] and result.get('completion_ok') else 'FAIL'} |"
        )
    measured = [
        r
        for r in results
        if not r["config"]["name"].startswith("controlled")
        and r["success"]
        and r["predicted_device_bytes"]
    ]
    if measured:
        errors = [
            abs(r["predicted_device_bytes"] - r["peak_device_total_bytes"])
            / r["peak_device_total_bytes"]
            * 100
            for r in measured
        ]
        deficits = [
            (r["peak_device_total_bytes"] - r["predicted_device_bytes"]) / MIB
            for r in measured
        ]
        rows += [
            "",
            f"Across {len(measured)} successful GPU configurations: maximum absolute error versus total peak {max(errors):.3f}%, mean {sum(errors) / len(errors):.3f}%, largest underestimate {max(deficits):.3f} MiB.",
        ]
        cpu_case = next(
            (r for r in results if r["config"]["name"] == "stories-cpu"), None
        )
        if cpu_case and cpu_case["success"] and cpu_case["peak_device_bytes"] == 0:
            rows.append("The CPU-only case allocates no additional Metal buffers.")
    for result in results:
        if (
            "estimation_error" in result
            and "allocation_limit_bytes" not in result["config"]
        ):
            message = " ".join(result.get("error", result["estimation_error"]).split())
            rows.append(
                f"Configuration `{result['config']['name']}`: estimation failed; "
                f"actual load {'succeeded' if result['success'] else 'failed'}. Native diagnostic: {message}"
            )
    limited = [r for r in results if "allocation_limit_bytes" in r["config"]]
    if limited:
        rows += [
            "",
            "## Controlled allocation-limit recovery",
            "",
            "These cases return nil from Metal buffer creation above a configured allocation budget.",
            "The advertised budget is deliberately higher. This injects a real allocator failure path without exhausting system RAM; it is not a physical OOM test.",
            "",
        ]
        for result in limited:
            rows.append(
                f"- {result['config']['name']}: success={result['success']}, "
                f"refused allocations={result.get('refused_allocations', 'unreported')}, "
                f"startup attempts={result.get('startup_attempts', 'failed')}, "
                f"inference={result.get('completion_ok', False)}, exit={result['exit_code']}."
            )
    rows += [
        "",
        "These measurements cover the listed local models on one Metal device. They do not establish CUDA/ROCm accuracy or guarantee peak usage for other workloads.",
        "",
    ]
    (output / "report.md").write_text("\n".join(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--controlled-oom", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("This benchmark measures Apple Metal; run it on macOS")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = [case for case in cases() if not args.case or case["name"] in args.case]
    if not selected:
        parser.error("No matching cases")
    if args.controlled_oom and not any(
        case["name"] == "llama-512" for case in selected
    ):
        parser.error("--controlled-oom requires the llama-512 case")
    results = []
    with tempfile.TemporaryDirectory(prefix="xllamacpp-memory-") as temporary:
        binary = Path(temporary) / "memory-accuracy"
        build_harness(binary, output / "build.log")
        for config in selected:
            if not Path(config["model"]).exists():
                raise FileNotFoundError(config["model"])
            results.append(run_case(binary, config, output))
            write_report(results, output)
        if args.controlled_oom:
            reference = next(
                result for result in results if result["config"]["name"] == "llama-512"
            )
            estimate = reference["predicted_device_bytes"]
            for automatic in [False, True]:
                config = {
                    **reference["config"],
                    "name": "controlled-auto" if automatic else "controlled-no-retry",
                    "n_gpu_layers": -1,
                    "fit_params": True,
                    "advertised_limit_bytes": estimate + GIB + 64 * MIB,
                    "allocation_limit_bytes": reference["baseline_bytes"]
                    + reference["loaded_device_bytes"]
                    - 8 * MIB,
                    "automatic_retry": automatic,
                }
                results.append(run_case(binary, config, output))
                write_report(results, output)
            config = {
                **reference["config"],
                "name": "controlled-weight-oom",
                "n_gpu_layers": -1,
                "fit_params": True,
                "automatic_retry": True,
                "advertised_limit_bytes": estimate + GIB + 64 * MIB,
                "allocation_limit_bytes": estimate // 2,
            }
            results.append(run_case(binary, config, output))
            write_report(results, output)
    hashes = {}
    for path in sorted({result["config"]["model"] for result in results}):
        with open(path, "rb") as model:
            digest = hashlib.sha256()
            while chunk := model.read(8 * MIB):
                digest.update(chunk)
            hashes[path] = digest.hexdigest()
    metadata = {
        "platform": platform.platform(),
        "estimator_sha256": hashlib.sha256(
            (ROOT / "src/xllamacpp/server-memory.h").read_bytes()
        ).hexdigest(),
        "harness_sha256": hashlib.sha256(
            (ROOT / "tests/native/server_memory_accuracy.mm").read_bytes()
        ).hexdigest(),
        "physical_memory_bytes": int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        ),
        "llama_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT / "thirdparty/llama.cpp", text=True
        ).strip(),
        "model_sha256": hashes,
        "results": results,
    }
    (output / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Report: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
