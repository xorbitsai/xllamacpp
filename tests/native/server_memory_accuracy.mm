#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>

#include "log.h"
#include "server-memory.h"
#include "server.h"

#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

using report_json = nlohmann::ordered_json;

static std::atomic<size_t> peak{0};
static std::atomic<size_t> refused{0};
static size_t allocation_limit = 0;
static size_t advertised_limit = 0;
static id<MTLDevice> metal_device;
static id<MTLBuffer> (*original_buffer)(id, SEL, NSUInteger, MTLResourceOptions);
static id<MTLBuffer> (*original_no_copy)(id, SEL, void *, NSUInteger, MTLResourceOptions, void (^)(void *, NSUInteger));

static void sample_memory() {
    const size_t current = metal_device.currentAllocatedSize;
    size_t previous = peak.load();
    while (current > previous && !peak.compare_exchange_weak(previous, current)) {
    }
}

static bool refuse_buffer(id device, NSUInteger length) {
    const size_t current = [(id<MTLDevice>)device currentAllocatedSize];
    if (allocation_limit && (current > allocation_limit || length > allocation_limit - current)) {
        ++refused;
        return true;
    }
    return false;
}

static id<MTLBuffer> tracked_buffer(id device, SEL selector, NSUInteger length, MTLResourceOptions options) {
    if (refuse_buffer(device, length)) {
        return nil;
    }
    id<MTLBuffer> buffer = original_buffer(device, selector, length, options);
    sample_memory();
    return buffer;
}

static id<MTLBuffer> tracked_no_copy(id device, SEL selector, void *bytes, NSUInteger length,
                                     MTLResourceOptions options, void (^deallocator)(void *, NSUInteger)) {
    if (refuse_buffer(device, length)) {
        return nil;
    }
    id<MTLBuffer> buffer = original_no_copy(device, selector, bytes, length, options, deallocator);
    sample_memory();
    return buffer;
}

static NSUInteger limited_working_set(id, SEL) { return advertised_limit; }

static void track_allocations() {
    const Class cls = object_getClass(metal_device);
    Method buffer = class_getInstanceMethod(cls, @selector(newBufferWithLength:options:));
    Method no_copy = class_getInstanceMethod(cls, @selector(newBufferWithBytesNoCopy:length:options:deallocator:));
    original_buffer =
        reinterpret_cast<decltype(original_buffer)>(method_setImplementation(buffer, (IMP)tracked_buffer));
    original_no_copy =
        reinterpret_cast<decltype(original_no_copy)>(method_setImplementation(no_copy, (IMP)tracked_no_copy));
    if (advertised_limit) {
        Method working_set = class_getInstanceMethod(cls, @selector(recommendedMaxWorkingSetSize));
        method_setImplementation(working_set, (IMP)limited_working_set);
    }
}

static bool collect_response(std::string &&data, void *destination) {
    *static_cast<std::string *>(destination) = std::move(data);
    return false;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        return 2;
    }
    @autoreleasepool {
        std::ifstream input(argv[1]);
        report_json config;
        input >> config;
        metal_device = MTLCreateSystemDefaultDevice();
        if (!metal_device) {
            std::cerr << "A Metal device is required\n";
            return 2;
        }
        allocation_limit = config.value("allocation_limit_bytes", size_t(0));
        advertised_limit = config.value("advertised_limit_bytes", size_t(0));
        track_allocations();
        common_init();
        common_log_set_verbosity_thold(0);
        llama_backend_init();

        common_params params;
        params.model.path = config.at("model").get<std::string>();
        params.n_ctx = config.value("n_ctx", 2048);
        params.n_batch = config.value("n_batch", 256);
        params.n_ubatch = config.value("n_ubatch", 128);
        params.n_parallel = config.value("n_parallel", 1);
        params.n_gpu_layers = config.value("n_gpu_layers", 99);
        params.fit_params = config.value("fit_params", false);
        params.warmup = true;
        params.verbosity = 0;
        params.port = 0;
        params.cache_ram_mib = 0;
        params.cpuparams.n_threads = 2;
        params.cpuparams_batch.n_threads = 2;
        params.flash_attn_type = static_cast<llama_flash_attn_type>(config.value("flash_attn", -1));
        if (config.value("quantized_kv", false)) {
            params.cache_type_k = GGML_TYPE_Q8_0;
            params.cache_type_v = GGML_TYPE_Q8_0;
        }
        params.tensor_buft_overrides.resize(llama_max_tensor_buft_overrides(), {nullptr, nullptr});

        report_json result = {{"config", config}, {"device", std::string([metal_device.name UTF8String])}};
        const size_t baseline = metal_device.currentAllocatedSize;
        result["baseline_bytes"] = baseline;
        result["predicted_device_bytes"] = nullptr;
        try {
            const auto estimate = xllamacpp::estimate_startup_memory(params);
            result["estimate"] = report_json::parse(estimate.report);
            size_t predicted_device = 0;
            for (const auto &row : result["estimate"]["buffers"]) {
                if (row["kind"] == "device") {
                    predicted_device += row["required_bytes"].get<size_t>();
                }
            }
            result["predicted_device_bytes"] = predicted_device;
        } catch (const std::exception &error) {
            result["estimation_error"] = error.what();
        }
        result["after_estimate_bytes"] = metal_device.currentAllocatedSize;
        std::cout << "XLLAMACPP_MEMORY_ESTIMATE=" << result.dump() << std::endl;
        peak = metal_device.currentAllocatedSize;

        std::atomic<bool> stop{false};
        std::thread monitor([&]() {
            while (!stop.load()) {
                sample_memory();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
        try {
            const bool automatic_retry = config.value("automatic_retry", false);
            xllamacpp::Server server(params, {}, nullptr, automatic_retry);
            sample_memory();
            result["loaded_device_bytes"] = metal_device.currentAllocatedSize - baseline;
            result["startup_peak_bytes"] = peak.load() - baseline;
            result["startup_attempts"] = server.startup_attempts();
            std::string prompt;
            const int repeats = config.value("prompt_repeats", std::min(96, params.n_ctx / params.n_parallel / 16));
            for (int i = 0; i < repeats; ++i) {
                prompt += "The little cat sat on the mat. ";
            }
            std::string response;
            const auto request =
                report_json({{"prompt", prompt}, {"n_predict", 16}, {"temperature", 0}, {"stream", false}}).dump();
            server.handle_completions(request, collect_response, &response, collect_response, &response);
            const auto completion = report_json::parse(response);
            result["completion_ok"] =
                !completion.contains("error") && completion.contains("choices") && !completion["choices"].empty();
            result["tokens_evaluated"] = completion.value("usage", report_json::object()).value("prompt_tokens", 0);
            result["tokens_predicted"] = completion.value("usage", report_json::object()).value("completion_tokens", 0);
            if (!result["completion_ok"].get<bool>()) {
                result["completion_error"] = completion;
            }
            sample_memory();
            result["inference_device_bytes"] = metal_device.currentAllocatedSize - baseline;
            result["peak_device_bytes"] = peak.load() - baseline;
            result["success"] = result["completion_ok"];
        } catch (const std::exception &error) {
            result["success"] = false;
            result["error"] = error.what();
            result["memory_error"] = dynamic_cast<const std::bad_alloc *>(&error) != nullptr;
            result["peak_device_bytes"] = peak.load() - baseline;
        }
        stop = true;
        monitor.join();
        result["peak_device_total_bytes"] = peak.load();
        result["peak_device_bytes"] = peak.load() - baseline;
        result["refused_allocations"] = refused.load();
        std::cout << "XLLAMACPP_MEMORY_BENCHMARK=" << result.dump() << std::endl;
        [metal_device release];
    }
}
