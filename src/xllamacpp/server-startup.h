#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace xllamacpp {

struct startup_result {
    bool        success    = false;
    bool        device_oom = false;
    bool        host_oom   = false;
    std::string stage      = "initialization";
    std::string detail;
    std::string errors;

    // Match allocator diagnostics, not arbitrary mentions of memory or filenames.
    void record_error(const std::string & text) {
        std::string lower = text;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
        const auto has = [&](const char * needle) {
            return lower.find(needle) != std::string::npos;
        };
        device_oom |= has("vk_error_out_of_device_memory") || has("erroroutofdevicememory") ||
                      has("cudaerrormemoryallocation") || has("hiperroroutofmemory") ||
                      ((has("cudamalloc failed:") || has("hipmalloc failed:")) && has("out of memory")) ||
                      (has("ggml_metal") && has("failed to allocate buffer, size =")) ||
                      (has("ggml_backend_metal_buffer_type_alloc_buffer:") &&
                       has("failed to allocate metal buffer of") && has("out of memory"));
        host_oom |= has("ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer of size");
        constexpr size_t limit = 8192;
        // Keep the tail, which usually includes the final allocation/load error.
        errors += text.substr(text.size() > limit ? text.size() - limit : 0);
        if (errors.size() > limit) {
            errors.erase(0, errors.size() - limit);
        }
    }

    std::string message() const {
        return stage + (detail.empty() ? " failed" : ": " + detail) + (errors.empty() ? "" : "\n" + errors);
    }
};

// This is a margin adjustment only; llama.cpp remains responsible for sizing.
inline bool increase_fit_margin(bool enabled, bool automatic_placement, std::vector<size_t> & margins, size_t extra) {
    if (!enabled || !automatic_placement || margins.empty() || extra == 0 ||
        std::any_of(margins.begin(), margins.end(),
                    [extra](size_t value) { return value > std::numeric_limits<size_t>::max() - extra; })) {
        return false;
    }
    for (auto & margin : margins) {
        margin += extra;
    }
    return true;
}

inline void validate_fit_retry_margins(const std::vector<size_t> & extra_margins) {
    size_t previous = 0;
    for (const auto margin : extra_margins) {
        if (margin <= previous) {
            throw std::invalid_argument("fit retry margins must be positive and strictly increasing");
        }
        previous = margin;
    }
}

// attempt() must finish cleanup before returning a failed result. next_margin()
// returns extra headroom relative to the original margins, or zero to stop.
template <typename Attempt, typename PrepareRetry, typename NextMargin>
startup_result start_with_retry_policy(Attempt      attempt,
                                       PrepareRetry prepare_retry,
                                       NextMargin   next_margin,
                                       unsigned &   attempts) {
    attempts   = 1;
    auto first = attempt();
    if (first.success) {
        return first;
    }
    auto result = first;
    while (result.stage == "model loading" && result.device_oom && !result.host_oom) {
        const size_t extra = next_margin();
        if (extra == 0 || !prepare_retry(extra)) {
            break;
        }
        ++attempts;
        result = attempt();
        if (result.success) {
            return result;
        }
    }
    if (attempts > 1) {
        result.detail += "\nFirst attempt: " + first.message();
    }
    return result;
}

template <typename Attempt, typename PrepareRetry>
startup_result start_with_retry(const std::vector<size_t> & extra_margins,
                                Attempt                     attempt,
                                PrepareRetry                prepare_retry,
                                unsigned &                  attempts) {
    validate_fit_retry_margins(extra_margins);
    size_t index = 0;
    return start_with_retry_policy(
        attempt, prepare_retry, [&]() { return index < extra_margins.size() ? extra_margins[index++] : size_t(0); },
        attempts);
}

inline size_t auto_fit_margin_limit() {
    constexpr size_t step = size_t(256) << 20;
    return static_cast<size_t>(std::min<uint64_t>(uint64_t(8) << 30, std::numeric_limits<size_t>::max() / step * step));
}

// A shortfall is only a hint: cleanup and refitting can hide the original gap.
// Always make progress, round up to 256 MiB, and cap extra headroom at 8 GiB.
inline size_t next_auto_fit_margin(size_t previous, size_t shortfall) {
    constexpr size_t step  = size_t(256) << 20;
    const size_t     limit = auto_fit_margin_limit();
    if (previous >= limit) {
        return 0;
    }
    const size_t doubled   = previous > limit / 2 ? limit : previous * 2;
    const size_t increment = shortfall >= limit - step ? limit : shortfall + step;
    const size_t increased = increment >= limit - previous ? limit : previous + increment;
    const size_t next      = std::max(doubled, increased);
    return next > limit - (step - 1) ? limit : (next + step - 1) / step * step;
}

template <typename Attempt, typename PrepareRetry, typename EstimateShortfall>
startup_result start_with_auto_retry(Attempt           attempt,
                                     PrepareRetry      prepare_retry,
                                     EstimateShortfall estimate_shortfall,
                                     unsigned &        attempts) {
    size_t   previous = 0;
    unsigned retries  = 0;
    return start_with_retry_policy(
        attempt, prepare_retry,
        [&]() {
            if (retries >= 4 || previous >= auto_fit_margin_limit()) {
                return size_t(0);
            }
            size_t shortfall = 0;
            try {
                shortfall = estimate_shortfall();
            } catch (...) {
                // Measurement is best effort; keep the bounded growth policy.
            }
            previous = next_auto_fit_margin(previous, shortfall);
            ++retries;
            return previous;
        },
        attempts);
}

// Cython's `except +` translates bad_alloc into Python MemoryError.
class startup_memory_error : public std::bad_alloc {
    std::string detail;
  public:
    explicit startup_memory_error(std::string message) : detail(std::move(message)) {}

    const char * what() const noexcept override { return detail.c_str(); }
};

}  // namespace xllamacpp
