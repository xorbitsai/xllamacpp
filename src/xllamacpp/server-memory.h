#pragma once

#include "common.h"
#include "fit.h"
#include "speculative.h"
#include <nlohmann/json.hpp>

#include <iomanip>
#include <sstream>

namespace xllamacpp {

struct startup_memory_estimate {
    std::string report;
    std::string summary;
    size_t      device_shortfall_bytes = 0;
};

inline startup_memory_estimate format_startup_memory(const std::vector<common_memory_breakdown_row> & rows,
                                                     uint32_t                                         n_ctx,
                                                     bool                                             auto_fit) {
    nlohmann::ordered_json report = {
        { "n_ctx",   n_ctx                                                                                           },
        { "basis",   auto_fit ? "auto_fit_after_cleanup" : "requested_configuration"                                 },
        { "scope",   "main model weights, context and compute buffers; excludes projector, draft model and adapters" },
        { "buffers", nlohmann::ordered_json::array()                                                                 },
    };
    std::ostringstream summary;
    summary << "\nEstimated memory required for main model (weights + context + compute)";
    if (auto_fit) {
        summary << "; auto-fit rerun after failed-load cleanup";
    }
    summary << ", n_ctx=" << n_ctx << ":" << std::fixed << std::setprecision(2);
    size_t           total     = 0;
    size_t           shortfall = 0;
    constexpr double MiB       = 1024.0 * 1024.0;
    for (const auto & row : rows) {
        const size_t required = row.model + row.context + row.compute;
        total += required;
        const bool             device = row.kind == common_memory_breakdown_row::DEVICE;
        nlohmann::ordered_json buffer = {
            { "kind",           device                                        ? "device" :
                      row.kind == common_memory_breakdown_row::HOST ? "host" :
                                                                      "buffer_type" },
            { "name",           row.name                                                                 },
            { "required_bytes", required                                                                 },
            { "model_bytes",    row.model                                                                },
            { "context_bytes",  row.context                                                              },
            { "compute_bytes",  row.compute                                                              },
            { "free_bytes",     nullptr                                                                  },
            { "total_bytes",    nullptr                                                                  },
        };
        summary << "\n  " << row.name << ": " << required / MiB << " MiB (weights " << row.model / MiB << ", context "
                << row.context / MiB << ", compute " << row.compute / MiB << ")";
        // Zero/zero is also used by backends with no memory-budget reporting.
        if (device && row.total > 0) {
            buffer["free_bytes"]  = row.free;
            buffer["total_bytes"] = row.total;
            const auto free       = static_cast<size_t>(std::max<int64_t>(0, row.free));
            if (required > free) {
                shortfall = std::max(shortfall, required - free);
            }
            summary << "; free " << row.free / MiB << " MiB, total " << row.total / MiB << " MiB";
        }
        report["buffers"].push_back(std::move(buffer));
    }
    report["required_bytes"]             = total;
    report["max_device_shortfall_bytes"] = shortfall;
    summary << "\nEstimate excludes projector, draft model, adapters and backend overhead; shared memory budgets may "
               "overlap.";
    return { report.dump(), summary.str(), shortfall };
}

// Only called after a recoverable OOM and after freeing the failed load.
// no_alloc measures buffer sizes without allocating model weights or KV buffers.
inline startup_memory_estimate estimate_startup_memory(common_params params) {
    // Match server_context's logits reservation for text generation/embeddings.
    // Multimodal output requirements depend on projector metadata.
    if (params.mmproj.path.empty()) {
        if (params.embedding ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
            params.n_outputs_max         = params.n_batch;
            params.n_outputs_max_per_seq = 1;
        } else {
            const auto limits    = common_speculative_get_output_limits(params.n_batch, params.n_parallel,
                                                                        common_speculative_n_max(&params.speculative));
            params.n_outputs_max = std::max<int32_t>(1, limits.total);
            params.n_outputs_max_per_seq = std::max<int32_t>(1, limits.per_seq);
        }
    }
    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);
    common_fit_apply(params, mparams, cparams);
    uint32_t n_ctx = 0;
    auto rows = common_fit_measure(params.model.get_paths(), &mparams, &cparams, nullptr, &n_ctx, GGML_LOG_LEVEL_ERROR);
    return format_startup_memory(rows, n_ctx, params.fit_params);
}

}  // namespace xllamacpp
