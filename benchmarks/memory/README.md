# Memory estimation validation

On 2026-11-14, the production estimator was compared with real Server startup
and inference on an Apple M2 Max with 32 GiB of unified memory. The matrix uses
four local GGUF models: Stories15M Q4_0, Qwen1.5 0.5B Q2_K, Llama 3.2 1B Q8_0,
and Stories15M MoE F16. It varies context size, GPU layer placement, parallel
slots, Flash Attention, and KV cache quantization.

- [Measurements and methodology](2026-11-14-metal-report.md)
- [Raw results, configurations, source hashes, and model SHA-256 hashes](2026-11-14-metal-results.json)
- [Baseline before correcting server output reservation](2026-11-14-metal-before.json)
- [Controlled weight-allocation failure log](2026-11-14-metal-weight-oom.txt)
- [Native crash backtrace](2026-11-14-metal-weight-oom-backtrace.txt)

For the 13 successful GPU configurations, the maximum absolute error against
the observed **total Metal allocation peak** was 2.742%; the largest
underestimate was 1.281 MiB. The total includes the 0.375 MiB of Metal allocations
already present before estimation. Against the additional allocations after
that baseline, the maximum error was 0.901%. The CPU-only case added no Metal
allocations. One intentionally exercised KV configuration was unsupported:
Stories has head dimension 48, which is incompatible with the q8_0 block size
32; both estimation and actual loading rejected it.

The baseline exposed a configuration mismatch: estimation reserved logits for a
full batch, whereas the Server reserves a smaller output set for generation.
The estimator now uses the same native speculative output-limit helpers as the
Server, with the embedding-mode exception. This reduced the Stories-512
estimate from 32.25 MiB to 18.93 MiB; its observed total peak was 19.47 MiB.

## Recovery evidence and limits

The harness also returns `nil` from Metal buffer creation when a controlled
allocation budget would be exceeded. It advertises a larger working-set budget
to emulate an optimistic memory report. It does not fill system RAM.

With the same context/compute allocation limit, startup without retries raised
the native memory exception after one rejected buffer allocation. Automatic
mode succeeded on attempt 2, evaluated 258 prompt tokens, and generated 16
tokens. The estimate showed no device shortfall after cleanup; recovery used
the fallback 256 MiB increase and native refitting. This demonstrates that the
bounded retry path works for that recoverable failure, not that it predicts the
exact missing margin.

A separate rejected **weight-buffer allocation** caused SIGSEGV in the vendored
Metal backend, in `ggml_metal_buffer_is_shared`, reached through
`ggml_backend_metal_buffer_shared_get_base` and `llama_model_loader::load_all_data`.
Automatic retries cannot recover from this process crash. The benchmark keeps
this failure in the raw results; its absence from the numerical accuracy
population is explicit. The native backend was not modified by this validation.

These are observations of Metal allocated bytes, not a measurement of physical
RAM residency or a guaranteed upper bound for other workloads. GPU command and
driver memory outside Metal's accounting can remain uncounted. CUDA, ROCm,
multimodal/projector allocations, draft models, and adapters were not validated.

## Reproduce

The local models listed in the raw configurations must be present under
`models/`. On macOS with Metal and the native build dependencies:

```sh
python setup.py build_ext --inplace
python scripts/benchmark_memory.py --output-dir /tmp/xllamacpp-memory-validation --controlled-oom
```

To run only the Llama 1B comparison and controlled failures:

```sh
python scripts/benchmark_memory.py --output-dir /tmp/xllamacpp-memory-validation --case llama-512 --controlled-oom
```

Each case runs in a fresh subprocess. The harness calls the production
`estimate_startup_memory` and `xllamacpp::Server`, records
`MTLDevice.currentAllocatedSize` immediately after buffer creation, and samples
every millisecond through startup and inference. It records both incremental
and total peaks. The injected weight-allocation case deliberately reproduces a
child-process crash. Configurations, logs, JSON results, and a Markdown report
are written to the requested output directory.
