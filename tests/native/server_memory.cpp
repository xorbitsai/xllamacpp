#include "server-memory.h"

#include <cassert>

using namespace xllamacpp;
using json = nlohmann::ordered_json;

int main() {
    constexpr size_t            MiB = 1024 * 1024;
    common_memory_breakdown_row gpu;
    gpu.name    = "GPU0";
    gpu.model   = 2048 * MiB;
    gpu.context = 512 * MiB;
    gpu.compute = 256 * MiB;
    gpu.free    = 1024 * MiB;
    gpu.total   = 4096 * MiB;
    common_memory_breakdown_row host;
    host.kind  = common_memory_breakdown_row::HOST;
    host.name  = "Host";
    host.model = 128 * MiB;

    const auto estimate = format_startup_memory({ gpu, host }, 4096, false);
    const auto report   = json::parse(estimate.report);
    assert(report["required_bytes"] == 2944 * MiB);
    assert(report["n_ctx"] == 4096);
    assert(report["basis"] == "requested_configuration");
    assert(estimate.device_shortfall_bytes == 1792 * MiB);
    assert(report["max_device_shortfall_bytes"] == 1792 * MiB);
    assert(report["buffers"].size() == 2);
    const auto & device = report["buffers"][0];
    assert(device["required_bytes"] == 2816 * MiB);
    assert(device["model_bytes"] == gpu.model);
    assert(device["context_bytes"] == gpu.context);
    assert(device["compute_bytes"] == gpu.compute);
    assert(device["free_bytes"] == gpu.free);
    assert(device["total_bytes"] == gpu.total);
    assert(report["buffers"][1]["kind"] == "host");
    assert(report["buffers"][1]["free_bytes"].is_null());
    assert(report["buffers"][1]["total_bytes"].is_null());
    assert(estimate.summary.find("GPU0: 2816.00 MiB") != std::string::npos);
    assert(estimate.summary.find("free 1024.00 MiB") != std::string::npos);
    assert(estimate.summary.find("Host: 128.00 MiB") != std::string::npos);

    gpu.free           = 0;
    gpu.total          = 0;
    const auto unknown = format_startup_memory({ gpu }, 128, true);
    assert(json::parse(unknown.report)["buffers"][0]["free_bytes"].is_null());
    assert(json::parse(unknown.report)["basis"] == "auto_fit_after_cleanup");
    assert(unknown.device_shortfall_bytes == 0);
    assert(unknown.summary.find("auto-fit rerun after failed-load cleanup") != std::string::npos);

    gpu.total = 4096 * MiB;
    assert(json::parse(format_startup_memory({ gpu }, 128, false).report)["buffers"][0]["free_bytes"] == 0);
    assert(format_startup_memory({ gpu }, 128, false).device_shortfall_bytes == 2816 * MiB);
    auto second_gpu = gpu;
    second_gpu.name = "GPU1";
    second_gpu.model += 128 * MiB;
    // The same extra margin applies to each device, so use the largest gap.
    assert(format_startup_memory({ gpu, second_gpu, host }, 128, false).device_shortfall_bytes == 2944 * MiB);
}
