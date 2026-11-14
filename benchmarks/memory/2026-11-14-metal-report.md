# Metal memory estimation measurements

The production estimator is compared with `MTLDevice.currentAllocatedSize` in fresh processes.
Buffer creation is instrumented to capture allocation peaks; a 1 ms sampler also runs through startup and inference.
Each successful case runs a real Server, prompt evaluation, and up to 16 generated tokens.

The peak is a Metal allocation high-water mark, not physical system RAM usage. CPU buffers are excluded from both sides.
Incremental peaks subtract the process's pre-estimation Metal allocation baseline. Total peaks include it, so backend initialization overhead remains visible.

| Configuration | Estimated MiB | Incremental peak MiB | Total peak MiB | Error vs total peak | Inference |
| --- | ---: | ---: | ---: | ---: | --- |
| stories-512 | 18.93 | 19.09 | 19.47 | -2.74% | pass |
| stories-4096 | 47.37 | 47.53 | 47.91 | -1.11% | pass |
| stories-partial | 19.70 | 19.88 | 20.25 | -2.74% | pass |
| stories-cpu | 0.00 | 0.00 | 0.38 | n/a | pass |
| stories-parallel4 | 51.40 | 51.56 | 51.94 | -1.03% | pass |
| stories-no-flash | 33.90 | 34.06 | 34.44 | -1.55% | pass |
| stories-q8-kv | n/a | n/a | n/a | n/a | estimation failed; load failed |
| qwen-1024 | 341.13 | 341.77 | 342.14 | -0.30% | pass |
| qwen-8192 | 1042.88 | 1043.77 | 1044.14 | -0.12% | pass |
| qwen-partial | 227.52 | 228.42 | 228.80 | -0.56% | pass |
| llama-512 | 1284.54 | 1285.06 | 1285.44 | -0.07% | pass |
| llama-4096 | 1403.68 | 1404.20 | 1404.58 | -0.06% | pass |
| llama-partial | 742.42 | 742.97 | 743.34 | -0.12% | pass |
| llama-q8-kv | 1344.56 | 1345.33 | 1345.70 | -0.09% | pass |
| moe-2048 | 70.36 | 70.53 | 70.91 | -0.77% | pass |

Across 13 successful GPU configurations: maximum absolute error versus total peak 2.742%, mean 0.867%, largest underestimate 1.281 MiB.
The CPU-only case allocates no additional Metal buffers.
Configuration `stories-q8-kv`: estimation failed; actual load failed. Native diagnostic: Failed to init server after 1 attempt(s): model loading failed llama_init_from_model: K cache type q8_0 with block size 32 does not divide n_embd_head_k=48

## Controlled allocation-limit recovery

These cases return nil from Metal buffer creation above a configured allocation budget.
The advertised budget is deliberately higher. This injects a real allocator failure path without exhausting system RAM; it is not a physical OOM test.

- controlled-no-retry: success=False, refused allocations=1, startup attempts=failed, inference=False, exit=0.
- controlled-auto: success=True, refused allocations=1, startup attempts=2, inference=True, exit=0.
- controlled-weight-oom: success=False, refused allocations=unreported, startup attempts=failed, inference=False, exit=-11.

These measurements cover the listed local models on one Metal device. They do not establish CUDA/ROCm accuracy or guarantee peak usage for other workloads.
