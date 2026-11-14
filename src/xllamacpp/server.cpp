#include "json-schema-to-grammar.h"
#include "server-connect.h"
#include "server-context.h"
#include "server-http.h"
#include "server-mcp.h"
#include "server-models.h"
#include "server-cors-proxy.h"
#include "server-stream.h"
#include "server-tools.h"
#include "server-startup.h"
#include "server-memory.h"

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <clocale>
#include <exception>
#include <future>
#include <mutex>
#include <thread>  // for std::thread::hardware_concurrency

// wrapper function that handles exceptions and logs errors
// this is to make sure handler_t never throws exceptions; instead, it returns an error response
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type  error;
        try {
            return func(req);
        } catch (const common_json_error & e) {
            // JSON parsing and request-shape errors are client errors, not internal failures.
            error   = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::invalid_argument & e) {
            // treat invalid_argument as invalid request (400)
            error   = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            // treat other exceptions as server error (500)
            error   = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error   = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res    = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status     = json_value(error_data, "code", 500);
            res->data       = safe_json_to_str({
                { "error", error_data }
            });
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

// adjust the params for the role; returns false on a configuration error
static bool server_prepare_params(common_params & params, server_role_type role_type) {
    if (role_type != SERVER_ROLE_ROUTER) {
        // validate batch size for embeddings
        // embeddings require all tokens to be processed in a single ubatch
        // see https://github.com/ggml-org/llama.cpp/issues/12836
        if (params.embedding && params.n_batch > params.n_ubatch) {
            SRV_WRN("embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", params.n_batch, params.n_ubatch);
            SRV_WRN("setting n_batch = n_ubatch = %d to avoid assertion failure\n", params.n_ubatch);
            params.n_batch = params.n_ubatch;
        }

        if (params.n_parallel < 0) {
            SRV_TRC("%s", "n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n");

            params.n_parallel = 4;
            params.kv_unified = true;
        }
    }

    // size the KV pool from --kv-unified-per-slot, unless the user pinned it with -c
    // or with -c 0 for max context
    const bool ctx_pool_auto_sized =
        params.kv_unified_per_slot > 0 && params.n_ctx == 0 && (uint32_t) params.fit_params_min_ctx != UINT32_MAX;

    if (ctx_pool_auto_sized) {
        params.n_ctx = params.n_parallel * params.kv_unified_per_slot;
        SRV_INF("--kv-unified-per-slot: sizing KV pool to n_parallel * kv_unified_per_slot = %d * %d = %d\n",
                params.n_parallel, params.kv_unified_per_slot, params.n_ctx);
    }

    // for consistency between server router mode and single-model mode, we set the same model name as alias
    auto model_name = params.model.get_name();
    if (params.model_alias.empty() && !model_name.empty()) {
        params.model_alias.insert(model_name);
    }

    // check early, so a missing llama-connect fails-fast
    if (params.server_connect) {
        const std::string reason = server_connect::unavailable_reason(params);
        if (!reason.empty()) {
            SRV_ERR("--connect is not available: %s\n", reason.c_str());
            return false;
        }
        params.server_connect_code = server_connect::resolve_code(params.server_connect_code);
    } else if (!params.server_connect_code.empty()) {
        // the code is only meaningful while the tunnel runs, and /props keys off it being set
        SRV_WRN("%s", "--connect-code is ignored without --connect\n");
        params.server_connect_code.clear();
    }

    return true;
}

// the API of every server; a router points the handlers in routes at its proxies before calling this
static void server_register_routes(server_http_context & ctx_http, server_routes & routes) {
    ctx_http.get("/health", ex_wrapper(routes.get_health));     // public endpoint (no API key check)
    ctx_http.get("/v1/health", ex_wrapper(routes.get_health));  // public endpoint (no API key check)
    ctx_http.get("/metrics", ex_wrapper(routes.get_metrics));
    ctx_http.get("/props", ex_wrapper(routes.get_props));
    ctx_http.post("/props", ex_wrapper(routes.post_props));
    ctx_http.get("/models", ex_wrapper(routes.get_models));
    ctx_http.get("/v1/models", ex_wrapper(routes.get_models));
    ctx_http.post("/completion", ex_wrapper(routes.post_completions));  // legacy
    ctx_http.post("/completions", ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions", ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions", ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions", ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions/control", ex_wrapper(routes.post_control));
    ctx_http.post("/v1/responses", ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/responses", ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/v1/audio/transcriptions", ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/audio/transcriptions", ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/tts", ex_wrapper(routes.post_tts));
    ctx_http.post("/v1/audio/speech", ex_wrapper(routes.post_speech_oai));
    ctx_http.post("/audio/speech", ex_wrapper(routes.post_speech_oai));
    ctx_http.ws("/v1/realtime", routes.ws_realtime);
    ctx_http.post("/v1/messages", ex_wrapper(routes.post_anthropic_messages));  // anthropic messages API
    ctx_http.post("/infill", ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding", ex_wrapper(routes.post_embeddings));            // legacy
    ctx_http.post("/embeddings", ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings", ex_wrapper(routes.post_embeddings_oai));
    ctx_http.post("/rerank", ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking", ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank", ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking", ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/systemone", ex_wrapper(routes.post_systemone));
    ctx_http.post("/tokenize", ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize", ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template", ex_wrapper(routes.post_apply_template));
    // token counting
    ctx_http.post("/chat/completions/input_tokens", ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/v1/chat/completions/input_tokens", ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/responses/input_tokens", ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/responses/input_tokens", ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/messages/count_tokens",
                  ex_wrapper(routes.post_anthropic_count_tokens));  // anthropic token counting
    // LoRA adapters hotswap
    ctx_http.get("/lora-adapters", ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters", ex_wrapper(routes.post_lora_adapters));
    // Save & load slots
    ctx_http.get("/slots", ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot", ex_wrapper(routes.post_slots));

    // Google Cloud Platform (Vertex AI) compat
    ctx_http.register_gcp_compat();
}

// resumable streaming, see server-stream.h: an instance binds the local session factories,
// a router binds proxies that resolve the owning child
static void server_register_stream_routes(server_http_context &                  ctx_http,
                                          const server_http_context::handler_t & stream_get,
                                          const server_http_context::handler_t & streams_lookup,
                                          const server_http_context::handler_t & stream_delete) {
    ctx_http.get("/v1/stream", ex_wrapper(stream_get));
    ctx_http.post("/v1/streams/lookup", ex_wrapper(streams_lookup));
    ctx_http.del("/v1/stream", ex_wrapper(stream_delete));
}

// the features behind a flag, and the warning about the ones that must not face untrusted clients
// returns false when one of them fails to start
static bool server_setup_features(server_http_context & ctx_http,
                                  const common_params & params,
                                  server_mcp &          mcp_mgr,
                                  server_tools &        tools,
                                  server_role_type      role_type) {
    // return 403 for disabled features
    server_http_context::handler_t res_403 = [](const server_http_req &) {
        auto res    = std::make_unique<server_http_res>();
        res->status = 403;
        res->data   = safe_json_to_str({
            { "error",
             {
                  { "message", "this feature is disabled" },
                  { "type", "feature_disabled" },
              } }
        });
        return res;
    };

    if (params.cors_origins == "*" && params.api_keys.empty()) {
        SRV_WRN("%s",
                "security: no API key is set and CORS allows all origins (see "
                "https://github.com/ggml-org/llama.cpp/pull/25655)\n");
    }

    // tools can run shell commands: do not expose them to the network without auth
    const bool tools_enabled = !params.server_tools.empty() || !params.mcp_servers_config.empty() ||
                               !params.mcp_servers_json.empty() || params.ui_mcp_proxy;
    const bool loopback_only = std::all_of(params.hostnames.begin(), params.hostnames.end(), [](const std::string & h) {
        return string_ends_with(h, ".sock") || h == "localhost" || h == "::1" || h == "[::1]" ||
               string_starts_with(h, "127.");
    });
    if (tools_enabled && !loopback_only && params.api_keys.empty() && !params.server_tools_allow_unauth) {
        SRV_ERR("%s",
                "security: server tools, MCP servers or MCP proxy are enabled on a non-loopback address without an API "
                "key\n");
        SRV_ERR("%s", "set --api-key, bind to 127.0.0.1, or pass --allow-unauthenticated-tools on a trusted network\n");
        return false;
    }

    std::vector<std::string> warn_names;
    if (role_type == SERVER_ROLE_ROUTER) {
        warn_names.push_back("router mode");
    }

    if (params.server_connect) {
        warn_names.push_back("peer-to-peer tunnel (llama-connect, experimental)");
    }

    // CORS proxy (EXPERIMENTAL, only used by the Web UI for MCP)
    if (params.ui_mcp_proxy) {
        ctx_http.get("/cors-proxy", ex_wrapper(proxy_handler_get));
        ctx_http.post("/cors-proxy", ex_wrapper(proxy_handler_post));
        warn_names.push_back("MCP proxy (experimental)");
    } else {
        ctx_http.get("/cors-proxy", ex_wrapper(res_403));
        ctx_http.post("/cors-proxy", ex_wrapper(res_403));
    }

    try {
        mcp_mgr.start(params);
    } catch (const std::exception & e) {
        SRV_ERR("MCP starting failed: %s\n", e.what());
        return false;
    }

    if (!params.server_tools.empty() || !mcp_mgr.empty()) {
        try {
            tools.setup(params.server_tools, mcp_mgr, params.server_tools_runtime);
        } catch (const std::exception & e) {
            SRV_ERR("tools setup failed: %s\n", e.what());
            return false;
        }
        ctx_http.get("/tools", ex_wrapper(tools.handle_get));
        ctx_http.post("/tools", ex_wrapper(tools.handle_post));
        if (!params.server_tools.empty()) {
            warn_names.push_back("server tools (experimental)");
        }
        if (!params.server_tools_runtime.empty()) {
            warn_names.push_back("tools runtime (experimental)");
        }
        if (!mcp_mgr.empty()) {
            warn_names.push_back("MCP servers (experimental)");
        }
    } else {
        ctx_http.get("/tools", ex_wrapper(res_403));
        ctx_http.post("/tools", ex_wrapper(res_403));
    }

    if (!warn_names.empty()) {
        std::string features;
        for (const auto & name : warn_names) {
            if (!features.empty()) {
                features += ", ";
            }
            features += name;
        }
        SRV_WRN("security: %s enabled - do not expose to untrusted environments\n", features.c_str());
    }

    return true;
}

// once the HTTP server listens: log the addresses and spawn llama-connect if asked, so the port is
// bound and the child startup health check reaches us. the destructor of connect_proc stops it
static bool server_on_listening(server_http_context & ctx_http,
                                const common_params & params,
                                server_connect &      connect_proc) {
    for (const auto & address : ctx_http.listening_addresses) {
        SRV_INF("listening on %s\n", address.c_str());
    }

    if (params.server_connect && !connect_proc.start(params, ctx_http.port)) {
        SRV_ERR("%s", "exiting due to llama-connect error\n");
        return false;
    }
    return true;
}

// Native fitting temporarily replaces the process-wide logger. Serialize startup
// only, and never leave a callback pointing at attempt-local storage.
static std::mutex                               startup_mutex;
static thread_local xllamacpp::startup_result * startup_diagnostics = nullptr;

static void startup_log_callback(ggml_log_level level, const char * text, void * user_data) {
    if (startup_diagnostics && level == GGML_LOG_LEVEL_ERROR) {
        try {
            startup_diagnostics->record_error(text);
        } catch (...) {
            // Logging must remain safe even when the host allocator is exhausted.
        }
    }
    common_log_default_callback(level, text, user_data);
}

static void init_impl(common_params &                           params,
                      server_context &                          ctx_server,
                      std::string &                             listening_address,
                      std::promise<xllamacpp::startup_result> & out,
                      xllamacpp::startup_result &               result,
                      std::unique_lock<std::mutex> &            startup_lock) {
    common_models_handler models_handler;
    if (!params.model.hf_repo.empty()) {
        result.stage   = "model metadata";
        models_handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
        if (common_models_handler_is_preset_repo(models_handler)) {
            common_models_handler_apply(models_handler, params);
        }
    }

    result.stage = "model parameters";
    common_params_print_info(params, true);
    if (!server_prepare_params(params, SERVER_ROLE_INSTANCE)) {
        return;
    }

    result.stage = "model download";
    common_models_handler_apply(models_handler, params);

    // These objects outlive the HTTP handlers that capture them. Their destructors
    // stop/join HTTP workers and shut down MCP, including on exceptional exits.
    server_mcp          mcp_mgr;
    server_routes       routes(params, ctx_server);
    server_tools        tools;
    server_http_context ctx_http;

    result.stage = "HTTP initialization";
    if (!ctx_http.init(params)) {
        result.detail = "failed to initialize HTTP server (check address/port and TLS settings)";
        return;
    }

    server_register_routes(ctx_http, routes);
    server_register_stream_routes(ctx_http, server_stream_make_get_handler(), server_stream_make_lookup_handler(),
                                  server_stream_make_delete_handler());

    result.stage = "server features";
    if (!server_setup_features(ctx_http, params, mcp_mgr, tools, SERVER_ROLE_INSTANCE)) {
        result.detail = "check API keys, tool authentication, and MCP configuration";
        return;
    }

    result.stage = "HTTP startup";
    if (!ctx_http.start()) {
        result.detail = "failed to bind/listen on HTTP address/port";
        return;
    }

    // Ignore earlier diagnostics when deciding whether model loading ran out of
    // device memory. HTTP/tools/download failures must never trigger a fit retry.
    result.stage = "model loading";
    result.errors.clear();
    result.device_oom = false;
    result.host_oom   = false;
    if (!ctx_server.load_model(params)) {
        return;
    }

    result.stage = "server metadata";
    result.errors.clear();
    result.device_oom = false;
    result.host_oom   = false;
    routes.update_meta(ctx_server);
    ctx_http.is_ready.store(true);
    SRV_INF("%s", "model loaded\n");

    result.stage = "server connection";
    server_connect connect_proc;
    if (!server_on_listening(ctx_http, params, connect_proc)) {
        return;
    }

    listening_address   = ctx_http.listening_addresses.front();
    result.success      = true;
    startup_diagnostics = nullptr;
    startup_lock.unlock();
    out.set_value(result);
    ctx_server.start_loop();

    ctx_http.stop();
    ctx_http.join();
    mcp_mgr.shutdown();

    // Rerank contexts cannot use common_memory_breakdown_print.
    auto * ll_ctx = ctx_server.get_llama_context();
    if (ll_ctx != nullptr && params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
        common_memory_breakdown_print(ll_ctx);
    }
}

static void init(common_params &                         params,
                 server_context &                        ctx_server,
                 std::string &                           listening_address,
                 std::promise<xllamacpp::startup_result> out) {
    std::unique_lock<std::mutex> startup_lock(startup_mutex);
    xllamacpp::startup_result    result;
    startup_diagnostics = &result;
    try {
        // Python owns the process locale and signal handlers.
        common_log_set_verbosity_thold(params.verbosity);
        common_init();
        llama_log_set(startup_log_callback, nullptr);
        server_stream_session_manager_start();
        SRV_INF("%s", "initializing ...\n");
        llama_backend_init();
        llama_numa_init(params.numa);
        SRV_INF("build_info: %s\n", llama_build_info());
        init_impl(params, ctx_server, listening_address, out, result, startup_lock);
    } catch (const std::bad_alloc & e) {
        result.host_oom = true;
        result.detail   = e.what();
    } catch (const std::exception & e) {
        result.detail = e.what();
        result.record_error(e.what());
    } catch (...) {
        result.detail = "unknown native exception";
    }
    startup_diagnostics = nullptr;
    server_stream_session_manager_stop();
    if (!result.success) {
        out.set_value(std::move(result));
    }
}

static void ggml_log_callback_default(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
    // if (level == GGML_LOG_LEVEL_INFO || level == GGML_LOG_LEVEL_ERROR) {
    //   fputs(text, stderr);
    //   fflush(stderr);
    // }
}

std::function<bool()> not_stop = [] {
    return false;
};

static std::vector<std::string> parse_oai_sse(const std::string & sse) {
    std::vector<std::string> out;

    std::size_t start = 0;
    while (start < sse.size()) {
        std::size_t end = sse.find('\n', start);
        if (end == std::string::npos) {
            break;
        }

        // Empty line = event separator, skip
        if (end > start) {
            // Guaranteed format: "data: <json>"
            out.emplace_back(sse.substr(start + 6, end - start - 6));
        }

        start = end + 1;
    }

    return out;
}

static void process_handler_response(server_http_res_ptr &               response,
                                     std::function<bool(std::string &&)> res_err,
                                     std::function<bool(std::string &&)> res_ok) {
    static const std::string sse_prefix("data: ");
    auto                     res = response->status == 200 ? res_ok : res_err;
    if (response->is_stream()) {
        std::string chunk;

        while (true) {
            const bool has_next = response->next(chunk);
            if (!chunk.empty() && chunk.size() >= sse_prefix.size()) {
                if (!has_next && chunk == "data: [DONE]\n\n") {
                    return;
                }
                auto parsed = parse_oai_sse(chunk);
                for (auto && json_str : parsed) {
                    if (res(std::move(json_str))) {
                        return;
                    }
                }
            }
            if (!has_next) {
                return;
            }
        }
    } else {
        res(std::move(response->data));
    }
}

#include "server.h"

namespace xllamacpp {

std::string get_system_info() {
    return llama_print_system_info();
}

std::vector<ggml_backend_dev_props> get_device_info() {
    ggml_log_set(ggml_log_callback_default, nullptr);

    const size_t dev_count = ggml_backend_dev_count();

    std::vector<ggml_backend_dev_props> result;
    std::vector<ggml_backend_dev_t>     devs;
    std::vector<ggml_backend_t>         backends;

    for (size_t i = 0; i < dev_count; ++i) {
        devs.push_back(ggml_backend_dev_get(i));

        ggml_backend_t backend = ggml_backend_dev_init(devs[i], NULL);
        GGML_ASSERT(backend != NULL);

        auto * reg = ggml_backend_dev_backend_reg(devs[i]);
        auto   ggml_backend_set_n_threads_fn =
            (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (ggml_backend_set_n_threads_fn) {
            ggml_backend_set_n_threads_fn(backend, std::thread::hardware_concurrency() / 2);
        }

        backends.push_back(backend);
    }

    for (size_t i = 0; i < dev_count; ++i) {
        // Put the backend to be tested in front so that it's prioritized:
        std::vector<ggml_backend_t> backends_modded = { backends[i] };
        backends_modded.insert(backends_modded.end(), backends.begin(), backends.end());

        ggml_backend_dev_props prop;
        ggml_backend_dev_get_props(devs[i], &prop);
        // Avoid crash when converting the prop struct to Python dict by Cython.
        if (prop.device_id == nullptr) {
            prop.device_id = "";
        }

        result.push_back(prop);
    }

    for (ggml_backend_t backend : backends) {
        ggml_backend_free(backend);
    }

    return result;
}

Server::Server(const common_params &       params,
               const std::vector<size_t> & fit_retry_margins,
               std::string *               startup_memory_report,
               bool                        auto_fit_retry) :
    _params(params) {
    validate_fit_retry_margins(fit_retry_margins);
    if (_params.model.path.empty()) {
        throw std::invalid_argument("params.model.path is required: xllamacpp disabled the router server feature");
    }

    // Keep a pristine snapshot: failed native loads may change their own params.
    auto       next_params = params;
    const auto attempt     = [&]() {
        _params = next_params;
        // Python skips the CLI's writable storage for automatic buffer overrides.
        _params.tensor_buft_overrides.resize(
            std::max(_params.tensor_buft_overrides.size(), llama_max_tensor_buft_overrides()), { nullptr, nullptr });
        _ctx_server = std::make_shared<server_context>();
        std::promise<startup_result> out;
        auto                         fut = out.get_future();
        _loop_thread =
            std::thread(init, std::ref(_params), std::ref(*_ctx_server), std::ref(_listening_address), std::move(out));
        auto result = fut.get();
        if (!result.success) {
            _loop_thread.join();
            // HTTP workers and routes have already been destroyed by init_impl.
            // Free every partial model/context/projector before remeasuring VRAM.
            _ctx_server.reset();
            llama_backend_free();
        }
        return result;
    };
    const bool automatic_placement =
        params.n_gpu_layers == -1 &&
        (params.split_mode == LLAMA_SPLIT_MODE_LAYER || params.split_mode == LLAMA_SPLIT_MODE_NONE) &&
        std::all_of(std::begin(params.tensor_split), std::end(params.tensor_split),
                    [](float value) { return value == 0; }) &&
        std::all_of(params.tensor_buft_overrides.begin(), params.tensor_buft_overrides.end(),
                    [](const llama_model_tensor_buft_override & value) { return value.pattern == nullptr; });
    const auto prepare_retry = [&](size_t extra) {
        // Every step is relative to the caller's margins, never cumulative.
        auto margins = params.fit_params_target;
        if (!increase_fit_margin(params.fit_params, automatic_placement, margins, extra)) {
            return false;
        }
        next_params.fit_params_target = std::move(margins);
        LOG_WRN("GPU allocation failed; retrying native auto-fit with an additional %.2f MiB per device\n",
                extra / (1024.0 * 1024.0));
        return true;
    };
    const auto estimate_shortfall = [&]() {
        std::lock_guard<std::mutex> lock(startup_mutex);
        return estimate_startup_memory(_params).device_shortfall_bytes;
    };
    const bool can_auto_retry =
        auto_fit_retry && params.fit_params && automatic_placement && !params.fit_params_target.empty();
    auto result = can_auto_retry ?
                      start_with_auto_retry(attempt, prepare_retry, estimate_shortfall, _startup_attempts) :
                      start_with_retry(fit_retry_margins, attempt, prepare_retry, _startup_attempts);
    if (!result.success) {
        auto message =
            "Failed to init server after " + std::to_string(_startup_attempts) + " attempt(s): " + result.message();
        if (result.device_oom || result.host_oom) {
            // Estimation must never replace the original allocation failure.
            try {
                std::lock_guard<std::mutex> lock(startup_mutex);
                const auto                  estimate = estimate_startup_memory(_params);
                message += estimate.summary;
                if (startup_memory_report) {
                    *startup_memory_report = estimate.report;
                }
            } catch (const std::exception & e) {
                message += "\nMemory estimate unavailable: " + std::string(e.what());
            } catch (...) {
                message += "\nMemory estimate unavailable";
            }
            throw startup_memory_error(message);
        }
        throw std::runtime_error(message);
    }

    try {
        _routes = std::make_shared<server_routes>(_params, *_ctx_server);
        _routes->update_meta(*_ctx_server);
    } catch (...) {
        _ctx_server->terminate();
        _loop_thread.join();
        _routes.reset();
        _ctx_server.reset();
        llama_backend_free();
        throw;
    }
}

Server::~Server() {
    _ctx_server->terminate();
    if (_loop_thread.joinable()) {
        _loop_thread.join();
    }
    _routes.reset();
    _ctx_server.reset();
    llama_backend_free();
}

unsigned Server::startup_attempts() const {
    return _startup_attempts;
}

std::string Server::listening_address() const {
    return _listening_address;
}

std::string Server::handle_metrics() {
    server_http_req req{ {}, {}, "", "", "", {}, not_stop };
    auto            res = _routes->get_metrics(req);
    return res->data;
}

std::string Server::handle_embeddings(const std::string & input_json_str) {
    server_http_req req{ {}, {}, "", "", input_json_str, {}, not_stop };
    auto            res = _routes->post_embeddings_oai(req);
    return res->data;
}

std::string Server::handle_rerank(const std::string & input_json_str) {
    server_http_req req{ {}, {}, "", "", input_json_str, {}, not_stop };
    auto            res = _routes->post_rerank(req);
    return res->data;
}

std::string Server::handle_systemone(const std::string & input_json_str) {
    server_http_req req{ {}, {}, "", "", input_json_str, {}, not_stop };
    auto            res = ex_wrapper(_routes->post_systemone)(req);
    return res->data;
}

void Server::handle_completions(const std::string & prompt_json_str,
                                Callback            res_err,
                                void *              py_cb_err,
                                Callback            res_ok,
                                void *              py_cb_ok) {
    server_http_req req{ {}, {}, "", "", prompt_json_str, {}, not_stop };
    auto            res = _routes->post_completions_oai(req);
    process_handler_response(
        res, [res_err, py_cb_err](std::string && err) { return res_err(std::move(err), py_cb_err); },
        [res_ok, py_cb_ok](std::string && ok) { return res_ok(std::move(ok), py_cb_ok); });
}

void Server::handle_chat_completions(const std::string & prompt_json_str,
                                     Callback            res_err,
                                     void *              py_cb_err,
                                     Callback            res_ok,
                                     void *              py_cb_ok) {
    server_http_req req{ {}, {}, "", "", prompt_json_str, {}, not_stop };
    auto            res = _routes->post_chat_completions(req);
    process_handler_response(
        res, [res_err, py_cb_err](std::string && err) { return res_err(std::move(err), py_cb_err); },
        [res_ok, py_cb_ok](std::string && ok) { return res_ok(std::move(ok), py_cb_ok); });
}

std::string json_schema_to_grammar_str(const std::string & schema_json_str) {
    try {
        auto schema = json::parse(schema_json_str);
        return json_schema_to_grammar(schema);
    } catch (const std::exception & e) {
        throw std::runtime_error(std::string("json_schema_to_grammar: ") + e.what());
    }
}

// Helper function to parse tensor buffer override strings
void parse_tensor_buffer_overrides(const std::string &                             value,
                                   std::vector<llama_model_tensor_buft_override> & overrides) {
    std::map<std::string, ggml_backend_buffer_type_t> buft_list;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * dev  = ggml_backend_dev_get(i);
        auto * buft = ggml_backend_dev_buffer_type(dev);
        if (buft) {
            buft_list[ggml_backend_buft_name(buft)] = buft;
        }
    }

    // Assignment semantics: replace any previous overrides instead of appending.
    overrides.clear();

    // An empty value clears the overrides: keep the vector empty (which maps to
    // a NULL pointer). string_split("") yields one empty token, which would
    // otherwise fail the '=' check below.
    if (value.empty()) {
        return;
    }

    for (const auto & override : string_split<std::string>(value, ',')) {
        std::string::size_type pos = override.find('=');
        if (pos == std::string::npos) {
            throw std::invalid_argument("invalid value");
        }
        std::string tensor_name = override.substr(0, pos);
        std::string buffer_type = override.substr(pos + 1);

        if (buft_list.find(buffer_type) == buft_list.end()) {
            printf("Available buffer types:\n");
            for (const auto & it : buft_list) {
                printf("  %s\n", ggml_backend_buft_name(it.second));
            }
            throw std::invalid_argument("unknown buffer type");
        }
        // keep strings alive and avoid leaking memory by storing them in a static
        // vector
        static std::list<std::string> buft_overrides;
        buft_overrides.push_back(tensor_name);
        overrides.push_back({ buft_overrides.back().c_str(), buft_list.at(buffer_type) });
    }

    // llama expects a NULL-terminated overrides array (a trailing entry with a
    // null pattern). Without this sentinel, common_model_params_to_llama() trips
    // the GGML_ASSERT "Tensor buffer overrides not terminated with empty pattern".
    // Only append the sentinel when at least one real override was provided so
    // that an empty value keeps the vector empty (which maps to a NULL pointer).
    if (!overrides.empty()) {
        overrides.push_back({ nullptr, nullptr });
    }
}

// Helper function to build tensor buffer override strings
void build_tensor_buffer_overrides(const std::vector<llama_model_tensor_buft_override> & overrides,
                                   std::string &                                         value) {
    std::map<ggml_backend_buffer_type_t, std::string> buft_list;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * dev  = ggml_backend_dev_get(i);
        auto * buft = ggml_backend_dev_buffer_type(dev);
        if (buft) {
            buft_list[buft] = ggml_backend_buft_name(buft);
        }
    }

    std::vector<std::string> parts;
    for (auto & override : overrides) {
        // Skip the NULL-terminator sentinel entry added by parse_tensor_buffer_overrides().
        if (override.pattern == nullptr) {
            continue;
        }
        std::string ov_str = std::string(override.pattern) + "=" + buft_list[override.buft];
        parts.emplace_back(ov_str);
    }

    value = string_join(parts, ",");
}

// Helper function to parse device list
std::vector<ggml_backend_dev_t> parse_device_list(const std::string & value) {
    std::vector<ggml_backend_dev_t> devices;
    auto                            dev_names = string_split<std::string>(value, ',');
    if (dev_names.empty()) {
        throw std::invalid_argument("no devices specified");
    }
    if (dev_names.size() == 1 && dev_names[0] == "none") {
        devices.push_back(nullptr);
    } else {
        for (const auto & device : dev_names) {
            auto * dev = ggml_backend_dev_by_name(device.c_str());
            if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                throw std::invalid_argument(string_format("invalid device: %s", device.c_str()));
            }
            devices.push_back(dev);
        }
        devices.push_back(nullptr);
    }
    return devices;
}

// Helper function to build device string from vector of ggml_backend_dev_t
std::string build_device_string(const std::vector<ggml_backend_dev_t> & devices) {
    if (devices.empty()) {
        return "";
    }
    if (devices.size() == 1 && devices[0] == nullptr) {
        return "";
    }
    std::vector<std::string> names;
    for (size_t i = 0; i < devices.size() - 1; ++i) {  // Skip the trailing nullptr
        if (devices[i]) {
            names.emplace_back(ggml_backend_dev_name(devices[i]));
        }
    }
    return string_join(names, ",");
}

}  // namespace xllamacpp
