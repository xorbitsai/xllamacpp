import pytest
from pytest import approx

import xllamacpp as xlc


def test_common_params_sampling():
    with pytest.raises(Exception, match="construct"):
        xlc.CommonParamsSampling()
    # assert params.seed == xlc.LLAMA_DEFAULT_SEED
    # assert params.n_prev == 64
    # assert params.n_probs == 0
    # assert params.min_keep == 0
    # assert params.top_k == 40
    # assert params.top_p == approx(0.95)
    # assert params.min_p == approx(0.05)
    # assert params.xtc_probability == 0.00
    # assert params.xtc_threshold == approx(0.10)
    # assert params.typ_p == approx(1.00)
    # assert params.temp == approx(0.80)
    # assert params.dynatemp_range == 0.00
    # assert params.dynatemp_exponent == approx(1.00)
    # assert params.penalty_last_n == 64
    # assert params.penalty_repeat == approx(1.00)
    # assert params.penalty_freq == 0.00
    # assert params.penalty_present == 0.00
    # assert params.dry_multiplier == 0.0
    # assert params.dry_base == approx(1.75)
    # assert params.dry_allowed_length == 2
    # assert params.dry_penalty_last_n == -1
    # assert params.mirostat == 0
    # assert params.mirostat_tau == approx(5.00)
    # assert params.mirostat_eta == approx(0.10)
    # assert params.ignore_eos is False
    # assert params.no_perf is False


def test_common_params():
    params = xlc.CommonParams()
    assert params.n_predict == -1
    assert params.n_ctx == 4096
    assert params.n_batch == 2048
    assert params.n_ubatch == 512
    assert params.n_keep == 0
    assert params.n_chunks == -1
    assert params.n_parallel == 1
    assert params.n_sequences == 1
    # assert params.p_split              ==   approx(0.1)
    assert params.n_gpu_layers == -1
    # assert params.n_gpu_layers_draft   ==    -1
    assert params.main_gpu == 0
    assert params.tensor_split == [0] * 128
    assert params.grp_attn_n == 1
    assert params.grp_attn_w == 512
    assert params.n_print == -1
    assert params.rope_freq_base == 0.0
    assert params.rope_freq_scale == 0.0
    assert params.yarn_ext_factor == approx(-1.0)
    assert params.yarn_attn_factor == approx(1.0)
    assert params.yarn_beta_fast == approx(32.0)
    assert params.yarn_beta_slow == approx(1.0)
    assert params.yarn_orig_ctx == 0
    assert params.defrag_thold == approx(0.1)

    assert params.cpuparams.n_threads == -1
    assert params.cpuparams.cpumask == [False] * xlc.GGML_MAX_N_THREADS
    assert params.cpuparams.mask_valid is False
    assert params.cpuparams.priority == xlc.GGML_SCHED_PRIO_NORMAL
    assert params.cpuparams.strict_cpu is False
    assert params.cpuparams.poll == 50

    # assert params.cpuparams_batch      ==
    # assert params.draft_cpuparams      ==
    # assert params.draft_cpuparams_batch ===

    # assert params.cb_eval             == nullptr;
    # assert params.cb_eval_user_data   == nullptr;

    assert params.numa == xlc.GGML_NUMA_STRATEGY_DISABLED
    assert params.split_mode == xlc.LLAMA_SPLIT_MODE_LAYER
    assert params.rope_scaling_type == xlc.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED
    assert params.pooling_type == xlc.LLAMA_POOLING_TYPE_UNSPECIFIED
    assert params.attention_type == xlc.LLAMA_ATTENTION_TYPE_UNSPECIFIED

    # common_sampler_params sparams

    assert params.model == ""
    # assert params.model_draft          == ""
    assert params.model_alias == ""
    assert params.model_url == ""
    assert params.hf_token == ""
    assert params.hf_repo == ""
    assert params.hf_file == ""
    assert params.prompt == ""
    assert params.prompt_file == ""
    assert params.path_prompt_cache == ""
    assert params.input_prefix == ""
    assert params.input_suffix == ""
    assert params.lookup_cache_static == ""
    assert params.lookup_cache_dynamic == ""
    assert params.logits_file == ""

    assert params.verbosity == 0
    assert params.control_vector_layer_start == -1
    assert params.control_vector_layer_end == -1
    assert params.ppl_stride == 0
    assert params.ppl_output_type == 0

    assert params.hellaswag is False
    assert params.hellaswag_tasks == 400
    assert params.winogrande is False
    assert params.winogrande_tasks == 0
    assert params.multiple_choice is False
    assert params.multiple_choice_tasks == 0
    assert params.kl_divergence is False
    assert params.usage is False
    assert params.use_color is False
    assert params.special is False
    assert params.interactive is False
    assert params.prompt_cache_all is False
    assert params.prompt_cache_ro is False
    assert params.escape is True
    assert params.multiline_input is False
    assert params.simple_io is False
    assert params.cont_batching is True
    assert params.flash_attn is False
    assert params.no_perf is False
    assert params.ctx_shift is True
    assert params.input_prefix_bos is False
    assert params.logits_all is False
    assert params.use_mmap is True
    assert params.use_mlock is False
    assert params.verbose_prompt is False
    assert params.display_prompt is True
    assert params.dump_kv_cache is False
    assert params.no_kv_offload is False
    assert params.warmup is True
    assert params.check_tensors is False

    assert params.cache_type_k == xlc.GGML_TYPE_F16
    assert params.cache_type_v == xlc.GGML_TYPE_F16

    assert params.mmproj == ""
    assert params.image == []

    assert params.embedding is False
    assert params.embd_normalize == 2
    assert params.embd_out == ""
    assert params.embd_sep == "\n"
    assert params.reranking is False

    assert params.port == 8080
    assert params.timeout_read == 600
    assert params.timeout_write == 600
    assert params.n_threads_http == -1
    assert params.n_cache_reuse == 0

    assert params.hostname == "127.0.0.1"
    assert params.public_path == ""
    assert params.chat_template == ""
    assert params.enable_chat_template is True

    assert params.api_keys == []
    assert params.ssl_file_key == ""
    assert params.ssl_file_cert == ""

    assert params.webui is True
    assert params.endpoint_slots is False
    assert params.endpoint_props is False
    assert params.endpoint_metrics is False

    assert params.log_json is False

    assert params.slot_save_path == ""

    assert params.slot_prompt_similarity == approx(0.5)

    assert params.is_pp_shared is False

    assert params.n_pp == []
    assert params.n_tg == []
    assert params.n_pl == []

    assert params.context_files == []
    assert params.chunk_size == 64
    assert params.chunk_separator == "\n"

    assert params.n_junk == 250
    assert params.i_pos == -1
    assert params.out_file == "imatrix.dat"

    assert params.n_out_freq == 10
    assert params.n_save_freq == 0
    assert params.i_chunk == 0

    assert params.process_output is False
    assert params.compute_ppl is True

    assert params.n_pca_batch == 100
    assert params.n_pca_iterations == 1000
    # assert params.cvector_dimre_method  == cy.DIMRE_METHOD_PCA
    # assert params.cvector_outfile       == "control_vector.gguf"
    # assert params.cvector_positive_file == "examples/cvector-generator/positive.txt"
    # assert params.cvector_negative_file == "examples/cvector-generator/negative.txt"

    # assert params.spm_infill            is False

    # assert params.lora_outfile          == "ggml-lora-merged-f16.gguf"

    # assert params.batched_bench_output_jsonl is False

    # ... rest not yet implemented
