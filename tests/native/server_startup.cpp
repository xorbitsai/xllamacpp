#include "server-startup.h"

#include <cassert>

using namespace xllamacpp;

static startup_result failure(const std::string & error, const std::string & stage = "model loading") {
    startup_result result;
    result.stage = stage;
    result.record_error(error);
    return result;
}

int main(int argc, char ** argv) {
    assert(argc == 2);
    const std::string scenario = argv[1];
    const std::string oom      = "ggml_backend_cuda_buffer_type_alloc_buffer: cudaMalloc failed: out of memory\n";
    if (scenario == "classification") {
        for (const auto & error : {
                 oom,
                 std::string("hipMalloc failed: out of memory"),
                 std::string("vk::Device::allocateMemory: VK_ERROR_OUT_OF_DEVICE_MEMORY"),
                 std::string("vk::Device::allocateMemory: ErrorOutOfDeviceMemory"),
                 std::string("ggml_metal_device_buffer_init: error: failed to allocate buffer, size = 512 MiB"),
             }) {
            assert(failure(error).device_oom);
        }
        for (const auto & error : {
                 "failed to open /models/out of memory.gguf",
                 "ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer",
                 "failed to allocate tensor metadata",
                 "cudaMalloc failed: invalid argument",
                 "failed to reserve graph",
             }) {
            assert(!failure(error).device_oom);
        }
        auto result = failure(oom);
        result.record_error(std::string(32000, 'x'));
        assert(result.errors.size() == 8192);
        assert(result.device_oom);
        try {
            throw startup_memory_error(result.message());
        } catch (const std::bad_alloc & e) {
            assert(std::string(e.what()).find("model loading") != std::string::npos);
        }
    } else if (scenario == "retry") {
        unsigned                  attempts        = 0;
        unsigned                  calls           = 0;
        unsigned                  preparations    = 0;
        bool                      live_allocation = false;
        const std::vector<size_t> base{ 100, 200 };
        const std::vector<size_t> schedule{ size_t(256) << 20, size_t(512) << 20, size_t(1) << 30, size_t(2) << 30 };
        auto                      margins = base;
        auto                      result  = start_with_retry(
            schedule,
            [&]() {
                // Model the lifetime of a partially loaded context. The controller
                // must see it destroyed before preparing/restarting the next load.
                assert(!live_allocation);
                live_allocation = true;
                struct allocation {
                    bool & live;
                    ~allocation() { live = false; }
                } context{ live_allocation };
                if (++calls < 4) {
                    return failure(oom);
                }
                assert(margins[0] == base[0] + schedule[2]);
                assert(margins[1] == base[1] + schedule[2]);
                startup_result loaded;
                loaded.success = true;
                return loaded;
            },
            [&](size_t extra) {
                assert(!live_allocation);
                assert(extra == schedule[preparations]);
                ++preparations;
                margins = base;
                return increase_fit_margin(true, true, margins, extra);
            },
            attempts);
        assert(result.success && attempts == 4 && calls == 4 && preparations == 3);
        assert(!live_allocation);
    } else if (scenario == "failures") {
        // Each failure mode must be bounded, including a second OOM.
        for (int mode = 0; mode < 7; ++mode) {
            unsigned calls        = 0;
            unsigned attempts     = 0;
            unsigned preparations = 0;
            auto     result       = start_with_retry(
                mode == 0 ? std::vector<size_t>{} : std::vector<size_t>{ 100, 200, 400 },
                [&]() {
                    ++calls;
                    if (mode == 1) {
                        return failure("invalid model");
                    }
                    if (mode == 2) {
                        return failure(oom, "HTTP initialization");
                    }
                    if (mode == 5) {
                        startup_result loaded;
                        loaded.success = true;
                        return loaded;
                    }
                    auto failed     = failure(oom);
                    failed.host_oom = mode == 6;
                    return failed;
                },
                [&](size_t) {
                    ++preparations;
                    return mode != 3;  // e.g. explicit placement or disabled fitting
                },
                attempts);
            const unsigned expected = mode == 4 ? 4 : 1;
            assert(calls == expected && attempts == expected);
            assert(preparations == (mode == 4 ? 3u : unsigned(mode == 3)));
            if (mode == 4) {
                assert(result.detail.find("First attempt:") != std::string::npos);
            }
        }
        unsigned   attempts = 0;
        unsigned   calls    = 0;
        const auto result   = start_with_retry(
            { 100, 200, 400 },
            [&]() {
                ++calls;
                return calls == 1 ? failure(oom) : failure("invalid model");
            },
            [](size_t) { return true; }, attempts);
        assert(!result.success && !result.device_oom && calls == 2 && attempts == 2);

        // Reject non-progressing schedules before starting a load.
        for (const auto & invalid : {
                 std::vector<size_t>{ 0 },
                 std::vector<size_t>{ 1, 1 },
                 std::vector<size_t>{ 2, 1 }
        }) {
            bool rejected = false;
            try {
                start_with_retry(
                    invalid,
                    [&]() {
                        ++calls;
                        return failure(oom);
                    },
                    [](size_t) { return true; }, attempts);
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            assert(rejected && calls == 2);
        }
    } else if (scenario == "margin") {
        const std::vector<size_t> original{ 0, 123456789, size_t(1) << 30 };
        auto                      margins = original;
        assert(!increase_fit_margin(false, true, margins, size_t(1) << 30));
        assert(!increase_fit_margin(true, false, margins, size_t(1) << 30));
        assert(margins == original);
        assert(increase_fit_margin(true, true, margins, size_t(1) << 30));
        for (size_t i = 0; i < margins.size(); ++i) {
            assert(margins[i] == original[i] + (size_t(1) << 30));
        }
        margins = { 100, std::numeric_limits<size_t>::max() };
        assert(!increase_fit_margin(true, true, margins, size_t(1) << 30));
        assert(margins[0] == 100);  // Never partially update on overflow.
        margins.clear();
        assert(!increase_fit_margin(true, true, margins, size_t(1) << 30));
    } else {
        return 1;
    }
}
