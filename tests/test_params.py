import pytest
from pytest import approx

import xllamacpp as xlc


def test_common_params_sampling():
    with pytest.raises(Exception, match="construct"):
        xlc.CommonParamsSampling()
    params = xlc.CommonParams()
    assert params.sampling.timing_per_token is False
    assert params.sampling.user_sampling_config == 0
    assert params.sampling.backend_sampling is False
    params.sampling.backend_sampling = True
    assert params.sampling.backend_sampling is True

    # Test new adaptive sampling fields
    assert params.sampling.adaptive_target == -1.0
    params.sampling.adaptive_target = 0.5
    assert params.sampling.adaptive_target == approx(0.5)

    assert params.sampling.adaptive_decay == approx(0.90)
    params.sampling.adaptive_decay = 0.95
    assert params.sampling.adaptive_decay == approx(0.95)

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


def test_enum_values():
    assert xlc.GGML_MAX_N_THREADS == 512
    assert xlc.GGML_ROPE_TYPE_VISION == 24
    assert xlc.ggml_sched_priority.GGML_SCHED_PRIO_REALTIME == 3
    assert xlc.ggml_numa_strategy.GGML_NUMA_STRATEGY_COUNT == 5
    assert xlc.ggml_type.GGML_TYPE_COUNT == 40
    assert xlc.ggml_backend_dev_type.GGML_BACKEND_DEVICE_TYPE_ACCEL == 3
    assert xlc.llama_rope_scaling_type.LLAMA_ROPE_SCALING_TYPE_MAX_VALUE == 3
    assert xlc.llama_pooling_type.LLAMA_POOLING_TYPE_RANK == 4
    assert xlc.llama_attention_type.LLAMA_ATTENTION_TYPE_NON_CAUSAL == 1
    assert xlc.llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_ENABLED == 1
    assert xlc.llama_split_mode.LLAMA_SPLIT_MODE_ROW == 2
    assert xlc.llama_model_kv_override_type.LLAMA_KV_OVERRIDE_TYPE_STR == 3
    assert xlc.dimre_method.DIMRE_METHOD_MEAN == 1
    assert xlc.common_conversation_mode.COMMON_CONVERSATION_MODE_AUTO == 2
    assert xlc.common_grammar_trigger_type.COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL == 3
    assert xlc.common_reasoning_format.COMMON_REASONING_FORMAT_DEEPSEEK == 3
    assert xlc.common_params_sampling_config.COMMON_PARAMS_SAMPLING_CONFIG_TEMP == 64


def test_common_params():
    params = xlc.CommonParams()
    assert params.n_predict == -1
    assert params.n_ctx == 0
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
    assert params.yarn_attn_factor == approx(-1.0)
    assert params.yarn_beta_fast == approx(-1.0)
    assert params.yarn_beta_slow == approx(-1.0)
    assert params.yarn_orig_ctx == 0

    assert params.cpuparams.n_threads == -1
    assert params.cpuparams.cpumask == [False] * xlc.GGML_MAX_N_THREADS
    assert params.cpuparams.mask_valid is False
    assert params.cpuparams.priority == xlc.ggml_sched_priority.GGML_SCHED_PRIO_NORMAL
    assert params.cpuparams.strict_cpu is False
    assert params.cpuparams.poll == 50

    # assert params.cpuparams_batch      ==
    # assert params.draft_cpuparams      ==
    # assert params.draft_cpuparams_batch ===

    # assert params.cb_eval             == nullptr;
    # assert params.cb_eval_user_data   == nullptr;

    assert params.numa == xlc.ggml_numa_strategy.GGML_NUMA_STRATEGY_DISABLED
    assert params.split_mode == xlc.llama_split_mode.LLAMA_SPLIT_MODE_LAYER
    assert (
        params.rope_scaling_type
        == xlc.llama_rope_scaling_type.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED
    )
    assert params.pooling_type == xlc.llama_pooling_type.LLAMA_POOLING_TYPE_UNSPECIFIED
    assert (
        params.attention_type
        == xlc.llama_attention_type.LLAMA_ATTENTION_TYPE_UNSPECIFIED
    )
    assert (
        params.flash_attn_type == xlc.llama_flash_attn_type.LLAMA_FLASH_ATTN_TYPE_AUTO
    )

    # common_sampler_params sparams

    assert params.model.path == ""
    assert params.model.url == ""
    assert params.model.hf_repo == ""
    assert params.model.hf_file == ""
    assert params.model.docker_repo == ""
    assert params.model.name == ""
    assert params.model_alias == ""
    assert params.hf_token == ""
    assert params.prompt == ""
    assert params.prompt_file == ""
    assert params.path_prompt_cache == ""
    assert params.input_prefix == ""
    assert params.input_suffix == ""
    assert params.speculative.lookup_cache_static == ""
    assert params.speculative.lookup_cache_dynamic == ""
    assert params.logits_file == ""

    # Test new debug properties
    assert params.logits_output_dir == "data"
    params.logits_output_dir = "/tmp/logits"
    assert params.logits_output_dir == "/tmp/logits"

    assert params.save_logits is False
    params.save_logits = True
    assert params.save_logits is True

    assert params.tensor_filter == []
    params.tensor_filter = ["tensor1", "tensor2"]
    assert params.tensor_filter == ["tensor1", "tensor2"]

    assert params.verbosity == 3
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
    assert params.no_perf is False
    assert params.show_timings is True
    assert params.ctx_shift is False
    assert params.swa_full is False
    assert params.kv_unified is False
    assert params.input_prefix_bos is False
    assert params.use_mmap is True
    assert params.use_direct_io is False
    params.use_direct_io = False
    assert params.use_direct_io is False
    assert params.use_mlock is False
    assert params.verbose_prompt is False
    assert params.display_prompt is True
    assert params.no_kv_offload is False
    assert params.warmup is True
    assert params.check_tensors is False
    assert params.no_op_offload is False
    assert params.no_extra_bufts is False
    assert params.no_host is False

    assert params.cache_type_k == xlc.ggml_type.GGML_TYPE_F16
    assert params.cache_type_v == xlc.ggml_type.GGML_TYPE_F16

    assert params.mmproj.path == ""
    assert params.mmproj_use_gpu is True
    assert params.no_mmproj is False
    assert params.image == []
    assert params.image_min_tokens == -1
    assert params.image_max_tokens == -1

    assert params.embedding is False
    assert params.embd_normalize == 2
    assert params.embd_out == ""
    assert params.embd_sep == "\n"

    assert params.port == 0
    assert params.timeout_read == 600
    assert params.timeout_write == 600
    assert params.n_threads_http == -1
    assert params.n_cache_reuse == 0
    assert params.cache_prompt is True
    params.cache_prompt = False
    assert params.cache_prompt is False
    assert params.n_ctx_checkpoints == 8
    assert params.cache_ram_mib == 8192

    assert params.hostname == "127.0.0.1"
    assert params.public_path == ""
    assert params.api_prefix == ""
    assert params.chat_template == ""
    assert params.use_jinja is True
    params.use_jinja = False
    assert params.use_jinja is False
    assert params.enable_chat_template is True
    assert (
        params.reasoning_format
        == xlc.common_reasoning_format.COMMON_REASONING_FORMAT_DEEPSEEK
    )
    assert params.prefill_assistant is True

    assert params.api_keys == []
    assert params.ssl_file_key == ""
    assert params.ssl_file_cert == ""

    params.default_template_kwargs = {"abc": "def"}
    assert params.default_template_kwargs == {"abc": "def"}

    assert params.webui is True
    assert params.endpoint_slots is True
    assert params.endpoint_props is False
    assert params.endpoint_metrics is False

    assert params.log_json is False

    assert params.slot_save_path == ""
    assert params.media_path == ""

    assert params.slot_prompt_similarity == approx(0.1)

    assert params.is_pp_shared is False
    assert params.is_tg_separate is False

    assert params.n_pp == []
    assert params.n_tg == []
    assert params.n_pl == []

    assert params.context_files == []
    assert params.chunk_size == 64
    assert params.chunk_separator == "\n"

    assert params.n_junk == 250
    assert params.i_pos == -1
    assert params.out_file == ""

    assert params.n_out_freq == 10
    assert params.n_save_freq == 0
    assert params.i_chunk == 0
    assert params.imat_dat == 0

    assert params.process_output is False
    assert params.compute_ppl is True
    assert params.parse_special is False

    assert params.n_pca_batch == 100
    assert params.n_pca_iterations == 1000

    # Test new fit_params fields
    assert params.fit_params is True
    params.fit_params = False
    assert params.fit_params is False
    for x in params.fit_params_target:
        assert x == 1024 * 1024 * 1024
    params.fit_params_target = [1024]
    assert params.fit_params_target == [1024]
    params.fit_params_target = [1024, 2048, 4096]
    assert params.fit_params_target == [1024, 2048, 4096]
    assert params.fit_params_min_ctx == 4096
    params.fit_params_min_ctx = 512
    assert params.fit_params_min_ctx == 512

    # Test new sleep_idle_seconds field
    assert params.sleep_idle_seconds == -1
    params.sleep_idle_seconds = 30
    assert params.sleep_idle_seconds == 30

    # Test new webui_config_json field
    assert params.webui_config_json == ""
    params.webui_config_json = '{"theme": "dark"}'
    assert params.webui_config_json == '{"theme": "dark"}'

    sp = params.sampling.samplers
    assert sp
    params.sampling.samplers = sp
    assert params.sampling.samplers == sp
    params.sampling.samplers = "top_k;top_p;min_p;temperature;dry;typ_p;xtc"
    assert params.sampling.samplers == "top_k;top_p;min_p;temperature;dry;typ_p;xtc"
    assert params.speculative.cache_type_k == xlc.ggml_type.GGML_TYPE_F16
    assert params.speculative.cache_type_v == xlc.ggml_type.GGML_TYPE_F16
    assert params.speculative.replacements == []
    params.speculative.replacements = [("a", "b")]
    assert params.speculative.replacements == [("a", "b")]

    # Test new speculative type field
    assert params.speculative.type == xlc.common_speculative_type.COMMON_SPECULATIVE_TYPE_NONE
    params.speculative.type = xlc.common_speculative_type.COMMON_SPECULATIVE_TYPE_DRAFT
    assert params.speculative.type == xlc.common_speculative_type.COMMON_SPECULATIVE_TYPE_DRAFT

    # Test new ngram-based speculative decoding fields
    assert params.speculative.ngram_size_n == 12
    params.speculative.ngram_size_n = 8
    assert params.speculative.ngram_size_n == 8

    assert params.speculative.ngram_size_m == 48
    params.speculative.ngram_size_m = 32
    assert params.speculative.ngram_size_m == 32

    assert params.speculative.ngram_check_rate == 1
    params.speculative.ngram_check_rate = 2
    assert params.speculative.ngram_check_rate == 2

    assert params.speculative.ngram_min_hits == 1
    params.speculative.ngram_min_hits = 3
    assert params.speculative.ngram_min_hits == 3

    # Test new p_split and p_min fields
    assert params.speculative.p_split == approx(0.1)
    params.speculative.p_split = 0.2
    assert params.speculative.p_split == approx(0.2)

    assert params.speculative.p_min == approx(0.75)
    params.speculative.p_min = 0.8
    assert params.speculative.p_min == approx(0.8)

    # Test mparams_dft (draft model params)
    assert params.speculative.mparams_dft.path == ""
    assert params.speculative.mparams_dft.hf_repo == ""
    assert params.speculative.mparams_dft.hf_file == ""

    assert params.cls_sep == "\t"
    assert params.offline is False
    assert params.reasoning_budget == -1

    assert params.diffusion.steps == 128
    params.diffusion.steps = 13
    assert params.diffusion.steps == 13
    assert params.diffusion.visual_mode is False
    params.diffusion.visual_mode = True
    assert params.diffusion.visual_mode is True
    assert params.diffusion.eps < 0.01
    params.diffusion.eps = 1.2
    assert 1.19 < params.diffusion.eps < 1.21
    assert params.diffusion.block_length == 0
    params.diffusion.block_length = 13
    assert params.diffusion.block_length == 13
    assert params.diffusion.algorithm == 4
    params.diffusion.algorithm = 1
    assert params.diffusion.algorithm == 1
    assert params.diffusion.alg_temp == 0.0
    params.diffusion.alg_temp = 1.1
    assert 1.09 < params.diffusion.alg_temp < 1.11
    assert params.diffusion.cfg_scale == 0.0
    params.diffusion.cfg_scale = 1.1
    assert 1.09 < params.diffusion.cfg_scale < 1.11
    assert params.diffusion.add_gumbel_noise is False
    params.diffusion.add_gumbel_noise = True
    assert params.diffusion.add_gumbel_noise is True

    assert params.tensor_buft_overrides == ""
    with pytest.raises(ValueError, match="unknown buffer type"):
        params.tensor_buft_overrides = (
            "blk\\.([0-3])\\.ffn_.*=GPU0,blk\\.4\\.ffn_(down|up)_exps\\..*=GPU0"
        )
    params.tensor_buft_overrides = (
        "blk\\.([0-3])\\.ffn_.*=CPU,blk\\.4\\.ffn_(down|up)_exps\\..*=CPU"
    )
    assert (
        params.tensor_buft_overrides
        == "blk\\.([0-3])\\.ffn_.*=CPU,blk\\.4\\.ffn_(down|up)_exps\\..*=CPU"
    )

    # assert params.cvector_dimre_method  == cy.DIMRE_METHOD_PCA
    # assert params.cvector_outfile       == "control_vector.gguf"
    # assert params.cvector_positive_file == "examples/cvector-generator/positive.txt"
    # assert params.cvector_negative_file == "examples/cvector-generator/negative.txt"

    # assert params.spm_infill            is False

    # assert params.lora_outfile          == "ggml-lora-merged-f16.gguf"

    # assert params.batched_bench_output_jsonl is False

    # ... rest not yet implemented


def test_json_schema_to_grammar():
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "score": {"type": "number"},
        },
        "required": ["answer"],
    }
    grammar = xlc.json_schema_to_grammar(schema)
    assert isinstance(grammar, str)
    assert grammar.strip()

    with pytest.raises(ValueError):
        xlc.json_schema_to_grammar("{not json}")


def test_lora_adapters():
    """Test CommonAdapterLoraInfo and CommonParams.lora_adapters property."""
    params = xlc.CommonParams()

    # Initially empty
    assert params.lora_adapters == []

    # Create and set adapters
    adapter1 = xlc.CommonAdapterLoraInfo("/path/lora1.gguf", 1.0)
    adapter2 = xlc.CommonAdapterLoraInfo("/path/lora2.gguf", 0.5)
    params.lora_adapters = [adapter1, adapter2]

    # Verify basic properties
    adapters = params.lora_adapters
    assert len(adapters) == 2
    assert adapters[0].path == "/path/lora1.gguf"
    assert adapters[0].scale == 1.0
    assert adapters[1].path == "/path/lora2.gguf"
    assert adapters[1].scale == approx(0.5)
    assert "lora1.gguf" in repr(adapters[0])

    # Test modifications affect underlying data
    adapter1.scale = 0.75
    assert params.lora_adapters[0].scale == approx(0.75)

    # Test default constructor
    default_adapter = xlc.CommonAdapterLoraInfo()
    assert default_adapter.path == ""
    assert default_adapter.scale == 1.0

    # Test dangling pointer safety - old wrappers become independent after reassignment
    old_adapters = params.lora_adapters
    params.lora_adapters = [xlc.CommonAdapterLoraInfo("/new.gguf", 2.0)]
    old_adapters[0].path = "/modified.gguf"  # Should not crash
    assert params.lora_adapters[0].path == "/new.gguf"  # Unchanged

    # Test set old wrappers
    params2 = xlc.CommonParams()
    old_adapters[0].path = "/other.gguf"
    old_adapters[0].scale = 3.0
    params2.lora_adapters = old_adapters
    assert params.lora_adapters[0].path == "/new.gguf"
    assert params2.lora_adapters[0].path == "/other.gguf"

    # Clear adapters
    params.lora_adapters = []
    assert params.lora_adapters == []
