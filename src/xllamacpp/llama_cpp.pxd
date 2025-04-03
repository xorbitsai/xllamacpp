# distutils: language=c++

from libc.stdint cimport int32_t, uint32_t, int64_t
from libcpp.string cimport string as std_string
from libcpp.vector cimport vector as std_vector
from libcpp.set cimport set as std_set


#------------------------------------------------------------------------------
# constants

cpdef enum:
    GGML_DEFAULT_N_THREADS = 4
    GGML_MAX_DIMS = 4
    GGML_MAX_N_THREADS = 16
    GGML_MAX_NAME = 64
    GGML_MAX_OP_PARAMS = 64
    GGML_MAX_SRC = 10

cpdef enum:
    GGML_ROPE_TYPE_NEOX   = 2
    GGML_ROPE_TYPE_MROPE  = 8
    GGML_ROPE_TYPE_VISION = 24


#------------------------------------------------------------------------------
# ggml.h

cdef extern from "ggml.h":

    cdef enum ggml_sched_priority:
        GGML_SCHED_PRIO_NORMAL
        GGML_SCHED_PRIO_MEDIUM
        GGML_SCHED_PRIO_HIGH
        GGML_SCHED_PRIO_REALTIME


    cdef enum ggml_numa_strategy:
        GGML_NUMA_STRATEGY_DISABLED   = 0
        GGML_NUMA_STRATEGY_DISTRIBUTE = 1
        GGML_NUMA_STRATEGY_ISOLATE    = 2
        GGML_NUMA_STRATEGY_NUMACTL    = 3
        GGML_NUMA_STRATEGY_MIRROR     = 4
        GGML_NUMA_STRATEGY_COUNT


    cdef enum ggml_type:
        GGML_TYPE_F32     = 0
        GGML_TYPE_F16     = 1
        GGML_TYPE_Q4_0    = 2
        GGML_TYPE_Q4_1    = 3
        # GGML_TYPE_Q4_2 = 4 support has been removed
        # GGML_TYPE_Q4_3 = 5 support has been removed
        GGML_TYPE_Q5_0    = 6
        GGML_TYPE_Q5_1    = 7
        GGML_TYPE_Q8_0    = 8
        GGML_TYPE_Q8_1    = 9
        GGML_TYPE_Q2_K    = 10
        GGML_TYPE_Q3_K    = 11
        GGML_TYPE_Q4_K    = 12
        GGML_TYPE_Q5_K    = 13
        GGML_TYPE_Q6_K    = 14
        GGML_TYPE_Q8_K    = 15
        GGML_TYPE_IQ2_XXS = 16
        GGML_TYPE_IQ2_XS  = 17
        GGML_TYPE_IQ3_XXS = 18
        GGML_TYPE_IQ1_S   = 19
        GGML_TYPE_IQ4_NL  = 20
        GGML_TYPE_IQ3_S   = 21
        GGML_TYPE_IQ2_S   = 22
        GGML_TYPE_IQ4_XS  = 23
        GGML_TYPE_I8      = 24
        GGML_TYPE_I16     = 25
        GGML_TYPE_I32     = 26
        GGML_TYPE_I64     = 27
        GGML_TYPE_F64     = 28
        GGML_TYPE_IQ1_M   = 29
        GGML_TYPE_BF16    = 30
        # GGML_TYPE_Q4_0_4_4 = 31 # support has been removed from gguf files
        # GGML_TYPE_Q4_0_4_8 = 32
        # GGML_TYPE_Q4_0_8_8 = 33
        GGML_TYPE_TQ1_0   = 34
        GGML_TYPE_TQ2_0   = 35
        GGML_TYPE_IQ4_NL_4_4 = 36
        # GGML_TYPE_IQ4_NL_4_4 = 36
        # GGML_TYPE_IQ4_NL_4_8 = 37
        # GGML_TYPE_IQ4_NL_8_8 = 38
        GGML_TYPE_COUNT = 39


    cdef enum ggml_op:
        GGML_OP_NONE = 0

        GGML_OP_DUP
        GGML_OP_ADD
        GGML_OP_ADD1
        GGML_OP_ACC
        GGML_OP_SUB
        GGML_OP_MUL
        GGML_OP_DIV
        GGML_OP_SQR
        GGML_OP_SQRT
        GGML_OP_LOG
        GGML_OP_SUM
        GGML_OP_SUM_ROWS
        GGML_OP_MEAN
        GGML_OP_ARGMAX
        GGML_OP_REPEAT
        GGML_OP_REPEAT_BACK
        GGML_OP_CONCAT
        GGML_OP_SILU_BACK
        GGML_OP_NORM # normalize
        GGML_OP_RMS_NORM
        GGML_OP_RMS_NORM_BACK
        GGML_OP_GROUP_NORM
        GGML_OP_L2_NORM

        GGML_OP_MUL_MAT
        GGML_OP_MUL_MAT_ID
        GGML_OP_OUT_PROD

        GGML_OP_SCALE
        GGML_OP_SET
        GGML_OP_CPY
        GGML_OP_CONT
        GGML_OP_RESHAPE
        GGML_OP_VIEW
        GGML_OP_PERMUTE
        GGML_OP_TRANSPOSE
        GGML_OP_GET_ROWS
        GGML_OP_GET_ROWS_BACK
        GGML_OP_DIAG
        GGML_OP_DIAG_MASK_INF
        GGML_OP_DIAG_MASK_ZERO
        GGML_OP_SOFT_MAX
        GGML_OP_SOFT_MAX_BACK
        GGML_OP_ROPE
        GGML_OP_ROPE_BACK
        GGML_OP_CLAMP
        GGML_OP_CONV_TRANSPOSE_1D
        GGML_OP_IM2COL
        GGML_OP_CONV_TRANSPOSE_2D
        GGML_OP_POOL_1D
        GGML_OP_POOL_2D
        GGML_OP_UPSCALE # nearest interpolate
        GGML_OP_PAD
        GGML_OP_ARANGE
        GGML_OP_TIMESTEP_EMBEDDING
        GGML_OP_ARGSORT
        GGML_OP_LEAKY_RELU

        GGML_OP_FLASH_ATTN_EXT
        GGML_OP_FLASH_ATTN_BACK
        GGML_OP_SSM_CONV
        GGML_OP_SSM_SCAN
        GGML_OP_WIN_PART
        GGML_OP_WIN_UNPART
        GGML_OP_GET_REL_POS
        GGML_OP_ADD_REL_POS

        GGML_OP_UNARY

        GGML_OP_MAP_UNARY
        GGML_OP_MAP_BINARY

        GGML_OP_MAP_CUSTOM1_F32
        GGML_OP_MAP_CUSTOM2_F32
        GGML_OP_MAP_CUSTOM3_F32

        GGML_OP_MAP_CUSTOM1
        GGML_OP_MAP_CUSTOM2
        GGML_OP_MAP_CUSTOM3

        GGML_OP_CROSS_ENTROPY_LOSS
        GGML_OP_CROSS_ENTROPY_LOSS_BACK

        GGML_OP_COUNT


    ctypedef struct ggml_backend_buffer_type:
        pass

    ctypedef struct ggml_backend_buffer:
        pass


    ctypedef ggml_backend_buffer_type * ggml_backend_buffer_type_t

    # -------------------------------------------------------------------------
    # n-dimensional tensor

    ctypedef struct ggml_tensor:
        ggml_type type

        ggml_backend_buffer * buffer

        int64_t ne[GGML_MAX_DIMS]  # number of elements
        size_t  nb[GGML_MAX_DIMS]  # stride in bytes:
                                   # nb[0] = ggml_type_size(type)
                                   # nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                                   # nb[i] = nb[i-1] * ne[i-1]

        # compute data
        ggml_op op

        # op params - allocated as int32_t for alignment
        int32_t op_params[16] # GGML_MAX_OP_PARAMS / sizeof(int32_t?)

        int32_t flags

        ggml_tensor * grad
        ggml_tensor * src[GGML_MAX_SRC]

        # source tensor and offset for views
        ggml_tensor * view_src
        size_t view_offs

        void * data

        char name[GGML_MAX_NAME]

        void * extra # extra things e.g. for ggml-cuda.cu

        # char padding[4]


#------------------------------------------------------------------------------
# ggml-backend.h


cdef extern from "ggml-backend.h":
    ctypedef bint (*ggml_backend_sched_eval_callback)(ggml_tensor * t, bint ask, void * user_data)

    ctypedef struct ggml_backend_device: pass

    ctypedef ggml_backend_device * ggml_backend_dev_t


#------------------------------------------------------------------------------
# llama.h

cdef extern from "llama.h":

    ctypedef int32_t llama_token

    cdef enum llama_rope_scaling_type:
        LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1
        LLAMA_ROPE_SCALING_TYPE_NONE        = 0
        LLAMA_ROPE_SCALING_TYPE_LINEAR      = 1
        LLAMA_ROPE_SCALING_TYPE_YARN        = 2
        LLAMA_ROPE_SCALING_TYPE_LONGROPE    = 3
        LLAMA_ROPE_SCALING_TYPE_MAX_VALUE   = LLAMA_ROPE_SCALING_TYPE_LONGROPE

    cdef enum llama_pooling_type:
        LLAMA_POOLING_TYPE_UNSPECIFIED = -1
        LLAMA_POOLING_TYPE_NONE = 0
        LLAMA_POOLING_TYPE_MEAN = 1
        LLAMA_POOLING_TYPE_CLS  = 2
        LLAMA_POOLING_TYPE_LAST = 3
        LLAMA_POOLING_TYPE_RANK = 4 # used by reranking models to attach the classification head to the graph
 
    cdef enum llama_attention_type:
        LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1
        LLAMA_ATTENTION_TYPE_CAUSAL      = 0
        LLAMA_ATTENTION_TYPE_NON_CAUSAL  = 1

    cdef enum llama_split_mode:
        LLAMA_SPLIT_MODE_NONE  = 0 # single GPU
        LLAMA_SPLIT_MODE_LAYER = 1 # split layers and KV across GPUs
        LLAMA_SPLIT_MODE_ROW   = 2 # split layers and KV across GPUs, use tensor parallelism if supported

    cdef enum llama_model_kv_override_type:
        LLAMA_KV_OVERRIDE_TYPE_INT
        LLAMA_KV_OVERRIDE_TYPE_FLOAT
        LLAMA_KV_OVERRIDE_TYPE_BOOL
        LLAMA_KV_OVERRIDE_TYPE_STR

    ctypedef struct llama_model_kv_override: # FLATTENED nested union enum
        llama_model_kv_override_type tag
        char key[128]
        int64_t val_i64
        double  val_f64
        bint    val_bool
        char    val_str[128]

    ctypedef struct llama_model_tensor_buft_override:
        const char * pattern
        ggml_backend_buffer_type_t buft

    ctypedef struct llama_logit_bias:
        llama_token token
        float bias


#------------------------------------------------------------------------------
# common.h

cdef extern from "common.h":

    ctypedef std_vector[llama_token] llama_tokens

    ctypedef struct llama_adapter_lora: pass

    cdef cppclass common_adapter_lora_info:
        std_string path
        float scale
        llama_adapter_lora *ptr

    ctypedef struct common_control_vector_load_info: pass

    # -------------------------------------------------------------------------
    # Build info

    cdef int LLAMA_BUILD_NUMBER
    cdef const char * LLAMA_COMMIT
    cdef const char * LLAMA_COMPILER
    cdef const char * LLAMA_BUILD_TARGET

    # -------------------------------------------------------------------------
    # CPU utils

    ctypedef struct cpu_params:
        int      n_threads
        bint     cpumask[GGML_MAX_N_THREADS] # CPU affinity mask.
        bint     mask_valid             # Default: any CPU
        ggml_sched_priority  priority   # Scheduling prio : (0 - normal, 1 - medium, 2 - high, 3 - realtime)
        bint     strict_cpu             # Use strict CPU placement
        uint32_t poll                   # Polling (busywait) level (0 - no polling, 100 - mostly polling)

    # -------------------------------------------------------------------------
    # Common params

    cdef enum llama_example:
        LLAMA_EXAMPLE_COMMON
        LLAMA_EXAMPLE_SPECULATIVE
        LLAMA_EXAMPLE_MAIN
        LLAMA_EXAMPLE_INFILL
        LLAMA_EXAMPLE_EMBEDDING
        LLAMA_EXAMPLE_PERPLEXITY
        LLAMA_EXAMPLE_RETRIEVAL
        LLAMA_EXAMPLE_PASSKEY
        LLAMA_EXAMPLE_IMATRIX
        LLAMA_EXAMPLE_BENCH
        LLAMA_EXAMPLE_SERVER
        LLAMA_EXAMPLE_CVECTOR_GENERATOR
        LLAMA_EXAMPLE_EXPORT_LORA
        LLAMA_EXAMPLE_LLAVA
        LLAMA_EXAMPLE_LOOKUP
        LLAMA_EXAMPLE_PARALLEL
        LLAMA_EXAMPLE_TTS
        LLAMA_EXAMPLE_COUNT

    cdef enum common_sampler_type:
        COMMON_SAMPLER_TYPE_NONE
        COMMON_SAMPLER_TYPE_TOP_K
        COMMON_SAMPLER_TYPE_TOP_P
        COMMON_SAMPLER_TYPE_MIN_P
        # COMMON_SAMPLER_TYPE_TFS_Z
        COMMON_SAMPLER_TYPE_TYPICAL_P
        COMMON_SAMPLER_TYPE_TEMPERATURE
        COMMON_SAMPLER_TYPE_XTC
        COMMON_SAMPLER_TYPE_INFILL
        COMMON_SAMPLER_TYPE_PENALTIES

    # dimensionality reduction methods, used by cvector-generator
    cdef enum dimre_method:
        DIMRE_METHOD_PCA
        DIMRE_METHOD_MEAN

    cdef enum common_conversation_mode:
        COMMON_CONVERSATION_MODE_DISABLED
        COMMON_CONVERSATION_MODE_ENABLED
        COMMON_CONVERSATION_MODE_AUTO

    cdef enum common_grammar_trigger_type:
        COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN
        COMMON_GRAMMAR_TRIGGER_TYPE_WORD
        COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN
        COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_START

    ctypedef struct common_grammar_trigger:
        common_grammar_trigger_type type
        std_string value
        bint at_start
        llama_token token

    # sampler parameters
    ctypedef struct common_params_sampling:
        uint32_t seed  # the seed used to initialize llama_sampler

        int32_t n_prev                     # number of previous tokens to remember
        int32_t n_probs                    # if greater than 0, output the probabilities of top n_probs tokens.
        int32_t min_keep                   # 0 = disabled, otherwise samplers should return at least min_keep tokens
        int32_t top_k                      # <= 0 to use vocab size
        float   top_p                      # 1.0 = disabled
        float   min_p                      # 0.0 = disabled
        float   xtc_probability            # 0.0 = disabled
        float   xtc_threshold              # > 0.5 disables XTC
        float   typ_p                      # typical_p, 1.0 = disabled
        float   temp                       # <= 0.0 to sample greedily, 0.0 to not output probabilities
        float   dynatemp_range             # 0.0 = disabled
        float   dynatemp_exponent          # controls how entropy maps to temperature in dynamic temperature sampler
        int32_t penalty_last_n             # last n tokens to penalize (0 = disable penalty, -1 = context size)
        float   penalty_repeat             # 1.0 = disabled
        float   penalty_freq               # 0.0 = disabled
        float   penalty_present            # 0.0 = disabled
        float   dry_multiplier             # 0.0 = disabled; DRY repetition penalty for tokens extending repetition:
        float   dry_base                   # 0.0 = disabled; multiplier * base ^ (length of sequence before token - allowed length)
        int32_t dry_allowed_length         # tokens extending repetitions beyond this receive penalty
        int32_t dry_penalty_last_n         # how many tokens to scan for repetitions (0 = disable penalty, -1 = context size)
        int32_t mirostat                   # 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
        float   top_n_sigma                # -1.0 = disabled
        float   mirostat_tau               # target entropy
        float   mirostat_eta               # learning rate
        bint    ignore_eos                 # ignore end-of-sentence
        bint    no_perf                    # disable performance metrics
        bint    timing_per_token

        std_vector[std_string] dry_sequence_breakers

        std_vector[common_sampler_type] samplers

        std_string grammar # optional BNF-like grammar to constrain sampling
        bint                                grammar_lazy
        std_vector[common_grammar_trigger]  grammar_triggers  # optional triggers (for lazy grammars)
        std_set[llama_token]                preserved_tokens

        std_vector[llama_logit_bias] logit_bias # logit biases to apply

        # print the parameters into a string
        # std_string print() const


    ctypedef struct common_params_model:
        std_string path          # model local path                                           // NOLINT
        std_string url           # model url to download                                      // NOLINT
        std_string hf_repo       # HF repo                                                    // NOLINT
        std_string hf_file       # HF file                                                    // NOLINT


    ctypedef struct common_params_speculative:
        std_vector[ggml_backend_dev_t] devices # devices to use for offloading
        int32_t n_ctx           # draft context size
        int32_t n_max           # maximum number of tokens to draft during speculative decoding
        int32_t n_min           # minimum number of draft tokens to use for speculative decoding
        int32_t n_gpu_layers    # number of layers to store in VRAM for the draft model (-1 - use default)
        float   p_split         # speculative decoding split probability
        float   p_min           # minimum speculative decoding probability (greedy)

        cpu_params cpuparams
        cpu_params cpuparams_batch
        common_params_model model


    ctypedef struct common_params_vocoder:
        common_params_model model
        std_string speaker_file # speaker file path                                      // NOLINT
        bint use_guide_tokens  # enable guide tokens to improve TTS accuracy            // NOLINT


    cdef enum common_reasoning_format:
        COMMON_REASONING_FORMAT_NONE
        COMMON_REASONING_FORMAT_DEEPSEEK  # Extract thinking tag contents and return as `message.reasoning_content`


    ctypedef struct common_params:
        llama_example curr_ex

        int32_t n_predict          # new tokens to predict
        int32_t n_ctx              # context size
        int32_t n_batch            # logical batch size for prompt processing (must be >=32 to use BLAS)
        int32_t n_ubatch           # physical batch size for prompt processing (must be >=32 to use BLAS)
        int32_t n_keep             # number of tokens to keep from initial prompt
        int32_t n_chunks           # max number of chunks to process (-1 = unlimited)
        int32_t n_parallel         # number of parallel sequences to decode
        int32_t n_sequences        # number of sequences to decode
        int32_t grp_attn_n         # group-attention factor
        int32_t grp_attn_w         # group-attention width
        int32_t n_print            # print token count every n tokens (-1 = disabled)
        float   rope_freq_base     # RoPE base frequency
        float   rope_freq_scale    # RoPE frequency scaling factor
        float   yarn_ext_factor    # YaRN extrapolation mix factor
        float   yarn_attn_factor   # YaRN magnitude scaling factor
        float   yarn_beta_fast     # YaRN low correction dim
        float   yarn_beta_slow     # YaRN high correction dim
        int32_t yarn_orig_ctx      # YaRN original context length
        float   defrag_thold       # KV cache defragmentation threshold

        std_vector[ggml_backend_dev_t] devices # devices to use for offloading
        int32_t n_gpu_layers       # number of layers to store in VRAM (-1 - use default)
        int32_t n_gpu_layers_draft # number of layers to store in VRAM for the draft model (-1 - use default)
        int32_t main_gpu           # the GPU that is used for scratch and small tensors
        float   tensor_split[128]  # how split tensors should be distributed across GPUs
        llama_split_mode        split_mode         # how to split the model across GPUs

        cpu_params cpuparams
        cpu_params cpuparams_batch

        ggml_backend_sched_eval_callback cb_eval
        void * cb_eval_user_data

        ggml_numa_strategy numa

        llama_rope_scaling_type rope_scaling_type
        llama_pooling_type      pooling_type       # pooling type for embeddings
        llama_attention_type    attention_type     # attention type for embeddings

        common_params_sampling sampling
        common_params_speculative speculative
        common_params_vocoder     vocoder
        common_params_model model

        std_string model_alias          # model alias
        std_string hf_token             # HF token
        std_string prompt               #
        std_string prompt_file          # store the external prompt file name
        std_string path_prompt_cache    # path to file for saving/loading prompt eval state
        std_string input_prefix         # string to prefix user inputs with
        std_string input_suffix         # string to suffix user inputs with
        std_string lookup_cache_static  # path of static ngram cache file for lookup decoding
        std_string lookup_cache_dynamic # path of dynamic ngram cache file for lookup decoding
        std_string logits_file          # file for saving *all* logits

        std_vector[std_string] in_files     # all input files
        std_vector[std_string] antiprompt   # strings upon which more user input is prompted (a.k.a. reverse prompts)
        std_vector[llama_model_kv_override] kv_overrides
        std_vector[llama_model_tensor_buft_override] tensor_buft_overrides


        bint lora_init_without_apply # only load lora to memory, but do not apply it to ctx (user can manually apply lora later using llama_lora_adapter_apply)
        std_vector[common_adapter_lora_info] lora_adapters # lora adapter path with user defined scale

        std_vector[common_control_vector_load_info] control_vectors # control vector with user defined scale

        int32_t verbosity
        int32_t control_vector_layer_start # layer range for control vector
        int32_t control_vector_layer_end   # layer range for control vector

        int32_t ppl_stride          # stride for perplexity calculations. If left at 0, the pre-existing approach will be used.
        int32_t ppl_output_type     # = 0 -> ppl output is as usual, = 1 -> ppl output is num_tokens, ppl, one per line

        bint   hellaswag            # compute HellaSwag score over random tasks from datafile supplied in prompt
        size_t hellaswag_tasks      # number of tasks to use when computing the HellaSwag score

        bint   winogrande           # compute Winogrande score over random tasks from datafile supplied in prompt
        size_t winogrande_tasks     # number of tasks to use when computing the Winogrande score. If 0, all tasks will be computed

        bint   multiple_choice      # compute TruthfulQA score over random tasks from datafile supplied in prompt
        size_t multiple_choice_tasks # number of tasks to use when computing the TruthfulQA score. If 0, all tasks will be computed

        bint   kl_divergence        # compute KL divergence

        bint usage                  # print usage
        bint completion             # print source-able completion script
        bint use_color              # use color to distinguish generations and inputs
        bint special                # enable special token output
        bint interactive            # interactive mode
        bint interactive_first      # wait for user input immediately
        bint prompt_cache_all       # save user input and generations to prompt cache
        bint prompt_cache_ro        # open the prompt cache read-only and do not update it

        bint escape                 # escape "\n", "\r", "\t", "\'", "\"", and "\\"
        bint multiline_input        # reverse the usage of `\`
        bint simple_io              # improves compatibility with subprocesses and limited consoles
        bint cont_batching          # insert new sequences for decoding on-the-fly
        bint flash_attn             # flash attention
        bint no_perf                # disable performance metric
        bint ctx_shift              # context shift on inifinite text generation

        bint input_prefix_bos       # prefix BOS to user inputs, preceding input_prefix
        bint logits_all             # return logits for all tokens in the batch
        bint use_mmap               # use mmap for faster loads
        bint use_mlock              # use mlock to keep model in memory
        bint verbose_prompt         # print prompt tokens before generation
        bint display_prompt         # print prompt before generation
        bint infill                 # use infill mode
        bint dump_kv_cache          # dump the KV cache contents for debugging purposes
        bint no_kv_offload          # disable KV offloading
        bint warmup                 # warmup run
        bint check_tensors          # validate tensor data
        bint single_turn            # single turn chat conversation

        ggml_type cache_type_k      # KV cache data type for the K
        ggml_type cache_type_v      # KV cache data type for the V

        common_conversation_mode conversation_mode

        # multimodal models (see examples/llava)
        common_params_model mmproj
        std_vector[std_string] image # path to image file(s)

        # embedding
        bint embedding              # get only sentence embedding
        int32_t embd_normalize      # normalisation for embeddings (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)
        std_string embd_out         # empty = default, "array" = [[],[]...], "json" = openai style, "json+" = same "json" + cosine similarity matrix
        std_string embd_sep         # separator of embeddings
        bint reranking              # enable reranking support on server

        # server params
        int32_t port                # server listens on this network port
        int32_t timeout_read        # http read timeout in seconds
        int32_t timeout_write       # http write timeout in seconds
        int32_t n_threads_http      # number of threads to process HTTP requests (TODO: support threadpool)
        int32_t n_cache_reuse       # min chunk size to reuse from the cache via KV shifting

        std_string hostname
        std_string public_path
        std_string chat_template
        bint use_jinja
        bint enable_chat_template

        common_reasoning_format reasoning_format

        std_vector[std_string] api_keys

        std_string ssl_file_key 
        std_string ssl_file_cert

        bint webui
        bint endpoint_slots
        bint endpoint_props
        bint endpoint_metrics

        bint log_json

        std_string slot_save_path

        float slot_prompt_similarity

        # batched-bench params
        bint is_pp_shared

        std_vector[int32_t] n_pp
        std_vector[int32_t] n_tg
        std_vector[int32_t] n_pl

        # retrieval params
        std_vector[std_string] context_files # context files to embed

        int32_t chunk_size      # chunk size for context embedding

        std_string chunk_separator # chunk separator for context embedding

        # passkey params
        int32_t n_junk      # number of times to repeat the junk text
        int32_t i_pos       # position of the passkey in the junk text

        # imatrix params
        int32_t n_out_freq       # output the imatrix every n_out_freq iterations
        int32_t n_save_freq      # save the imatrix every n_save_freq iterations
        int32_t i_chunk          # start processing from this chunk

        bint process_output      # collect data for the output tensor
        bint compute_ppl         # whether to compute perplexity

        # cvector-generator params
        int n_pca_batch
        int n_pca_iterations
        dimre_method cvector_dimre_method
        std_string cvector_positive_file
        std_string cvector_negative_file

        bint spm_infill

        # batched-bench params
        bint batched_bench_output_jsonl
    
        # common params
        std_string out_file      # output filename for all example programs
