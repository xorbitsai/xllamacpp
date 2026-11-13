#pragma once

#include <algorithm>
#include <cctype>
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
                      (has("ggml_metal") && has("failed to allocate buffer, size ="));
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

// attempt() must finish cleanup before returning a failed result.
template <typename Attempt, typename PrepareRetry>
startup_result start_with_retry(const std::vector<size_t> & extra_margins,
                                Attempt                     attempt,
                                PrepareRetry                prepare_retry,
                                unsigned &                  attempts) {
    validate_fit_retry_margins(extra_margins);
    attempts   = 1;
    auto first = attempt();
    if (first.success || extra_margins.empty()) {
        return first;
    }
    auto result = first;
    for (const auto extra : extra_margins) {
        if (result.stage != "model loading" || !result.device_oom || result.host_oom || !prepare_retry(extra)) {
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

// Cython's `except +` translates bad_alloc into Python MemoryError.
class startup_memory_error : public std::bad_alloc {
    std::string detail;
  public:
    explicit startup_memory_error(std::string message) : detail(std::move(message)) {}

    const char * what() const noexcept override { return detail.c_str(); }
};

}  // namespace xllamacpp
