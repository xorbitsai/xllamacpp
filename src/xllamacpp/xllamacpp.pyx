# distutils: language = c++
# cython: profile=False
# cython: embedsignature = True
# cython: language_level = 3
# cython: c_string_encoding = default

"""
xllamacpp: a thin cython wrapper of llama.cpp
"""
from libc.stdint cimport int32_t, uint32_t
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr, make_shared
from cython.operator cimport dereference as deref

cimport llama_cpp
from server cimport CServer


# constants
# -----------------------------------------------------------------------------

LLAMA_DEFAULT_SEED = 0xFFFFFFFF

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


# build info
# -----------------------------------------------------------------------------

BUILD_INFO = {
    'build_number': llama_cpp.LLAMA_BUILD_NUMBER,
    'commit': llama_cpp.LLAMA_COMMIT.decode(),
    'compiler': llama_cpp.LLAMA_COMPILER.decode(),
    'build_target': llama_cpp.LLAMA_BUILD_TARGET.decode(),
}

# enums
# -----------------------------------------------------------------------------

cpdef enum ggml_log_level:
    GGML_LOG_LEVEL_NONE  = 0
    GGML_LOG_LEVEL_INFO  = 1
    GGML_LOG_LEVEL_WARN  = 2
    GGML_LOG_LEVEL_ERROR = 3
    GGML_LOG_LEVEL_DEBUG = 4
    GGML_LOG_LEVEL_CONT  = 5

cpdef enum llama_vocab_type:
    LLAMA_VOCAB_TYPE_NONE # For models without vocab
    LLAMA_VOCAB_TYPE_SPM  # LLaMA tokenizer based on byte-level BPE with byte fallback
    LLAMA_VOCAB_TYPE_BPE  # GPT-2 tokenizer based on byte-level BPE
    LLAMA_VOCAB_TYPE_WPM  # BERT tokenizer based on WordPiece
    LLAMA_VOCAB_TYPE_UGM  # T5 tokenizer based on Unigram
    LLAMA_VOCAB_TYPE_RWKV # RWKV tokenizer based on greedy tokenization

cpdef enum llama_rope_type:
    LLAMA_ROPE_TYPE_NONE   = -1
    LLAMA_ROPE_TYPE_NORM   = 0
    LLAMA_ROPE_TYPE_NEOX   = GGML_ROPE_TYPE_NEOX
    LLAMA_ROPE_TYPE_MROPE  = GGML_ROPE_TYPE_MROPE
    LLAMA_ROPE_TYPE_VISION = GGML_ROPE_TYPE_VISION

cpdef enum llama_token_attr:
    LLAMA_TOKEN_ATTR_UNDEFINED    = 0
    LLAMA_TOKEN_ATTR_UNKNOWN      = 1 << 0
    LLAMA_TOKEN_ATTR_UNUSED       = 1 << 1
    LLAMA_TOKEN_ATTR_NORMAL       = 1 << 2
    LLAMA_TOKEN_ATTR_CONTROL      = 1 << 3 # SPECIAL?
    LLAMA_TOKEN_ATTR_USER_DEFINED = 1 << 4
    LLAMA_TOKEN_ATTR_BYTE         = 1 << 5
    LLAMA_TOKEN_ATTR_NORMALIZED   = 1 << 6
    LLAMA_TOKEN_ATTR_LSTRIP       = 1 << 7
    LLAMA_TOKEN_ATTR_RSTRIP       = 1 << 8
    LLAMA_TOKEN_ATTR_SINGLE_WORD  = 1 << 9

cpdef enum ggml_numa_strategy:
    GGML_NUMA_STRATEGY_DISABLED   = 0
    GGML_NUMA_STRATEGY_DISTRIBUTE = 1
    GGML_NUMA_STRATEGY_ISOLATE    = 2
    GGML_NUMA_STRATEGY_NUMACTL    = 3
    GGML_NUMA_STRATEGY_MIRROR     = 4
    GGML_NUMA_STRATEGY_COUNT

cpdef enum ggml_type:
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


cpdef enum ggml_sched_priority:
    GGML_SCHED_PRIO_NORMAL
    GGML_SCHED_PRIO_MEDIUM
    GGML_SCHED_PRIO_HIGH
    GGML_SCHED_PRIO_REALTIME

cpdef enum llama_ftype:
    LLAMA_FTYPE_ALL_F32              = 0
    LLAMA_FTYPE_MOSTLY_F16           = 1
    LLAMA_FTYPE_MOSTLY_Q4_0          = 2
    LLAMA_FTYPE_MOSTLY_Q4_1          = 3
    # LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4  # tok_embeddings.weight and output.weight are F16
    # LLAMA_FTYPE_MOSTLY_Q4_2       = 5     # support has been removed
    # LLAMA_FTYPE_MOSTLY_Q4_3       = 6     # support has been removed
    LLAMA_FTYPE_MOSTLY_Q8_0          = 7
    LLAMA_FTYPE_MOSTLY_Q5_0          = 8
    LLAMA_FTYPE_MOSTLY_Q5_1          = 9
    LLAMA_FTYPE_MOSTLY_Q2_K          = 10
    LLAMA_FTYPE_MOSTLY_Q3_K_S        = 11
    LLAMA_FTYPE_MOSTLY_Q3_K_M        = 12
    LLAMA_FTYPE_MOSTLY_Q3_K_L        = 13
    LLAMA_FTYPE_MOSTLY_Q4_K_S        = 14
    LLAMA_FTYPE_MOSTLY_Q4_K_M        = 15
    LLAMA_FTYPE_MOSTLY_Q5_K_S        = 16
    LLAMA_FTYPE_MOSTLY_Q5_K_M        = 17
    LLAMA_FTYPE_MOSTLY_Q6_K          = 18
    LLAMA_FTYPE_MOSTLY_IQ2_XXS       = 19
    LLAMA_FTYPE_MOSTLY_IQ2_XS        = 20
    LLAMA_FTYPE_MOSTLY_Q2_K_S        = 21
    LLAMA_FTYPE_MOSTLY_IQ3_XS        = 22
    LLAMA_FTYPE_MOSTLY_IQ3_XXS       = 23
    LLAMA_FTYPE_MOSTLY_IQ1_S         = 24
    LLAMA_FTYPE_MOSTLY_IQ4_NL        = 25
    LLAMA_FTYPE_MOSTLY_IQ3_S         = 26
    LLAMA_FTYPE_MOSTLY_IQ3_M         = 27
    LLAMA_FTYPE_MOSTLY_IQ2_S         = 28
    LLAMA_FTYPE_MOSTLY_IQ2_M         = 29
    LLAMA_FTYPE_MOSTLY_IQ4_XS        = 30
    LLAMA_FTYPE_MOSTLY_IQ1_M         = 31
    LLAMA_FTYPE_MOSTLY_BF16          = 32
    # LLAMA_FTYPE_MOSTLY_Q4_0_4_4      = 33, # removed from gguf files, use Q4_0 and runtime repack
    # LLAMA_FTYPE_MOSTLY_Q4_0_4_8      = 34, # removed from gguf files, use Q4_0 and runtime repack
    # LLAMA_FTYPE_MOSTLY_Q4_0_8_8      = 35, # removed from gguf files, use Q4_0 and runtime repack
    LLAMA_FTYPE_MOSTLY_TQ1_0         = 36 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_TQ2_0         = 37 # except 1d tensors
    LLAMA_FTYPE_GUESSED              = 1024

cpdef enum llama_rope_scaling_type:
    LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1
    LLAMA_ROPE_SCALING_TYPE_NONE        = 0
    LLAMA_ROPE_SCALING_TYPE_LINEAR      = 1
    LLAMA_ROPE_SCALING_TYPE_YARN        = 2
    LLAMA_ROPE_SCALING_TYPE_LONGROPE    = 3
    LLAMA_ROPE_SCALING_TYPE_MAX_VALUE   = LLAMA_ROPE_SCALING_TYPE_LONGROPE

cpdef enum llama_pooling_type:
    LLAMA_POOLING_TYPE_UNSPECIFIED = -1
    LLAMA_POOLING_TYPE_NONE = 0
    LLAMA_POOLING_TYPE_MEAN = 1
    LLAMA_POOLING_TYPE_CLS  = 2
    LLAMA_POOLING_TYPE_LAST = 3
    LLAMA_POOLING_TYPE_RANK = 4 # used by reranking models to attach the classification head to the graph

cpdef enum llama_attention_type:
    LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1
    LLAMA_ATTENTION_TYPE_CAUSAL      = 0
    LLAMA_ATTENTION_TYPE_NON_CAUSAL  = 1

cpdef enum llama_split_mode:
    LLAMA_SPLIT_MODE_NONE  = 0
    LLAMA_SPLIT_MODE_LAYER = 1
    LLAMA_SPLIT_MODE_ROW   = 2

cpdef enum common_sampler_type:
    COMMON_SAMPLER_TYPE_NONE        = 1
    COMMON_SAMPLER_TYPE_TOP_K       = 2
    COMMON_SAMPLER_TYPE_TOP_P       = 3
    COMMON_SAMPLER_TYPE_MIN_P       = 4
    # COMMON_SAMPLER_TYPE_TFS_Z     = 5
    COMMON_SAMPLER_TYPE_TYPICAL_P   = 6
    COMMON_SAMPLER_TYPE_TEMPERATURE = 7
    COMMON_SAMPLER_TYPE_XTC         = 8
    COMMON_SAMPLER_TYPE_INFILL      = 9
    COMMON_SAMPLER_TYPE_PENALTIES   = 10


cdef class LlamaLogitBias:
    cdef llama_cpp.llama_logit_bias *p
    cdef object owner

    @staticmethod
    cdef LlamaLogitBias from_ptr(llama_cpp.llama_logit_bias *p, object owner):
        cdef LlamaLogitBias wrapper = LlamaLogitBias.__new__(LlamaLogitBias)
        wrapper.p = p
        wrapper.owner = owner
        return wrapper

    def __init__(self):
        raise Exception(f"Can't construct an instance of {type(self).__name__}")

    @property
    def token(self) -> int:
        """token token"""
        return self.p.token

    @token.setter
    def token(self, int value):
        self.p.token = value

    @property
    def bias(self) -> float:
        """bias"""
        return self.p.bias

    @bias.setter
    def bias(self, float value):
        self.p.bias = value


cdef class CommonParamsSampling:
    cdef llama_cpp.common_params_sampling *p
    cdef object owner

    @staticmethod
    cdef CommonParamsSampling from_ptr(llama_cpp.common_params_sampling *params, object owner):
        cdef CommonParamsSampling wrapper = CommonParamsSampling.__new__(CommonParamsSampling)
        wrapper.p = params
        wrapper.owner = owner
        return wrapper

    def __init__(self):
        raise Exception(f"Can't construct an instance of {type(self).__name__}")

    def print(self) -> str:
        """print the parameters into a string"""
        return ( 
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f" % (
                self.penalty_last_n, self.penalty_repeat, self.penalty_freq, self.penalty_present,
                self.dry_multiplier, self.dry_base, self.dry_allowed_length, self.dry_penalty_last_n,
                self.top_k, self.top_p, self.min_p, self.xtc_probability, self.xtc_threshold, self.typ_p, self.temp,
                self.mirostat, self.mirostat_eta, self.mirostat_tau)
        )

    @property
    def seed(self) -> int:
        """the seed used to initialize llama_sampler."""
        return self.p.seed

    @seed.setter
    def seed(self, uint32_t value):
        self.p.seed = value

    @property
    def n_prev(self) -> int:
        """number of previous tokens to remember"""
        return self.p.n_prev

    @n_prev.setter
    def n_prev(self, int32_t value):
        self.p.n_prev = value

    @property
    def n_probs(self) -> int:
        """if greater than 0, output the probabilities of top n_probs tokens."""
        return self.p.n_probs

    @n_probs.setter
    def n_probs(self, int32_t value):
        self.p.n_probs = value

    @property
    def min_keep(self) -> int:
        """if greater than 0, output the probabilities of top min_keep tokens."""
        return self.p.min_keep

    @min_keep.setter
    def min_keep(self, int32_t value):
        self.p.min_keep = value

    @property
    def top_k(self) -> int:
        """<= 0 to use vocab size."""
        return self.p.top_k

    @top_k.setter
    def top_k(self, int32_t value):
        self.p.top_k = value

    @property
    def top_p(self) -> float:
        """1.0 = disabled"""
        return self.p.top_p

    @top_p.setter
    def top_p(self, float value):
        self.p.top_p = value

    @property
    def min_p(self) -> float:
        """0.0 = disabled"""
        return self.p.min_p

    @min_p.setter
    def min_p(self, float value):
        self.p.min_p = value

    @property
    def xtc_probability(self) -> float:
        """0.0 = disabled"""
        return self.p.xtc_probability

    @xtc_probability.setter
    def xtc_probability(self, float value):
        self.p.xtc_probability = value

    @property
    def xtc_threshold(self) -> float:
        """> 0.5 disables XTC"""
        return self.p.xtc_threshold

    @xtc_threshold.setter
    def xtc_threshold(self, float value):
        self.p.xtc_threshold = value

    # @property
    # def tfs_z(self) -> float:
    #     """1.0 = disabled"""
    #     return self.p.tfs_z

    # @tfs_z.setter
    # def tfs_z(self, float value):
    #     self.p.tfs_z = value

    @property
    def typ_p(self) -> float:
        """typical_p, 1.0 = disabled"""
        return self.p.typ_p

    @typ_p.setter
    def typ_p(self, float value):
        self.p.typ_p = value

    @property
    def temp(self) -> float:
        """<= 0.0 to sample greedily, 0.0 to not output probabilities"""
        return self.p.temp

    @temp.setter
    def temp(self, float value):
        self.p.temp = value

    @property
    def dynatemp_range(self) -> float:
        """0.0 = disabled"""
        return self.p.dynatemp_range

    @dynatemp_range.setter
    def dynatemp_range(self, float value):
        self.p.dynatemp_range = value

    @property
    def dynatemp_exponent(self) -> float:
        """controls how entropy maps to temperature in dynamic temperature sampler"""
        return self.p.dynatemp_exponent

    @dynatemp_exponent.setter
    def dynatemp_exponent(self, float value):
        self.p.dynatemp_exponent = value

    @property
    def penalty_last_n(self) -> int:
        """last n tokens to penalize (0 = disable penalty, -1 = context size)"""
        return self.p.penalty_last_n

    @penalty_last_n.setter
    def penalty_last_n(self, int value):
        self.p.penalty_last_n = value

    @property
    def penalty_repeat(self) -> float:
        """1.0 = disabled"""
        return self.p.penalty_repeat

    @penalty_repeat.setter
    def penalty_repeat(self, float value):
        self.p.penalty_repeat = value

    @property
    def penalty_freq(self) -> float:
        """0.0 = disabled"""
        return self.p.penalty_freq

    @penalty_freq.setter
    def penalty_freq(self, float value):
        self.p.penalty_freq = value

    @property
    def penalty_present(self) -> float:
        """0.0 = disabled"""
        return self.p.penalty_present

    @penalty_present.setter
    def penalty_present(self, float value):
        self.p.penalty_present = value

    @property
    def dry_multiplier(self) -> float:
        """0.0 = disabled

        DRY repetition penalty for tokens extending repetition
        """
        return self.p.dry_multiplier

    @dry_multiplier.setter
    def dry_multiplier(self, float value):
        self.p.dry_multiplier = value

    @property
    def dry_base(self) -> float:
        """0.0 = disabled

        multiplier * base ^ (length of sequence before token - allowed length)
        """
        return self.p.dry_base

    @dry_base.setter
    def dry_base(self, float value):
        self.p.dry_base = value

    @property
    def dry_allowed_length(self) -> int:
        """tokens extending repetitions beyond this receive penalty"""
        return self.p.dry_allowed_length

    @dry_allowed_length.setter
    def dry_allowed_length(self, int value):
        self.p.dry_allowed_length = value

    @property
    def dry_penalty_last_n(self) -> int:
        """how many tokens to scan for repetitions (0 = disable penalty, -1 = context size)"""
        return self.p.dry_penalty_last_n

    @dry_penalty_last_n.setter
    def dry_penalty_last_n(self, int value):
        self.p.dry_penalty_last_n = value

    @property
    def mirostat(self) -> int:
        """0 = disabled, 1 = mirostat, 2 = mirostat 2.0"""
        return self.p.mirostat

    @mirostat.setter
    def mirostat(self, int value):
        self.p.mirostat = value

    @property
    def mirostat_tau(self) -> float:
        """target entropy"""
        return self.p.mirostat_tau

    @mirostat_tau.setter
    def mirostat_tau(self, float value):
        self.p.mirostat_tau = value

    @property
    def mirostat_eta(self) -> float:
        """learning rate"""
        return self.p.mirostat_eta

    @mirostat_eta.setter
    def mirostat_eta(self, float value):
        self.p.mirostat_eta = value

    # @property
    # def penalize_nl(self) -> bool:
    #     """consider newlines as a repeatable token"""
    #     return self.p.penalize_nl

    # @penalize_nl.setter
    # def penalize_nl(self, bint value):
    #     self.p.penalize_nl = value

    @property
    def ignore_eos(self) -> bool:
        """ignore end-of-sentence"""
        return self.p.ignore_eos

    @ignore_eos.setter
    def ignore_eos(self, bint value):
        self.p.ignore_eos = value

    @property
    def no_perf(self) -> bool:
        """disable performance metrics"""
        return self.p.no_perf

    @no_perf.setter
    def no_perf(self, bint value):
        self.p.no_perf = value

    @property
    def samplers(self) -> list[common_sampler_type]:
        """get/set sampler types
        
        std_vector[common_sampler_type] samplers
        """
        return self.p.samplers

    @samplers.setter
    def samplers(self, value: list[common_sampler_type]):
        self.p.samplers = value

    @property
    def grammar(self) -> str:
        """optional BNF-like grammar to constrain sampling"""
        return self.p.grammar

    @grammar.setter
    def grammar(self, str value):
        self.p.grammar = value

    @property
    def logit_bias(self) -> list[LlamaLogitBias]:
        """logit biases to apply
        
        std_vector[llama_logit_bias] logit_bias
        """
        result = []
        for i in range(self.p.logit_bias.size()):
            result.append(LlamaLogitBias.from_ptr(&self.p.logit_bias[i], self))
        return result

    @logit_bias.setter
    def logit_bias(self, elems: list[LlamaLogitBias]):
        cdef vector[llama_cpp.llama_logit_bias] vec
        for elem in elems:
            vec.push_back(elem.ptr[0])
        self.p.logit_bias = vec



cdef class CpuParams:
    cdef llama_cpp.cpu_params *p
    cdef object owner

    @staticmethod
    cdef CpuParams from_ptr(llama_cpp.cpu_params *params, object owner):
        cdef CpuParams wrapper = CpuParams.__new__(CpuParams)
        wrapper.p = params
        wrapper.owner = owner
        return wrapper

    def __init__(self):
        raise Exception(f"Can't construct an instance of {type(self).__name__}")

    @property
    def n_threads(self) -> int:
        """number of threads."""
        return self.p.n_threads

    @n_threads.setter
    def n_threads(self, value: int):
        self.p.n_threads = value

    @property
    def cpumask(self) -> list[bool]:
        """CPU affinity mask: mask of cpu cores (all-zeros means use default affinity settings)
        
        cpumask[GGML_MAX_N_THREADS] is (by default) of size 16
        """
        res = []
        for i in range(GGML_MAX_N_THREADS):
            res.append(<bint>self.p.cpumask[i])
        return res

    @cpumask.setter
    def cpumask(self, values: list[bool]):
        assert len(values) == GGML_MAX_N_THREADS
        for i in range(GGML_MAX_N_THREADS):
            self.p.cpumask[i] = <bint>values[i]

    @property
    def mask_valid(self) -> bool:
        """Default: any CPU."""
        return self.p.mask_valid

    @mask_valid.setter
    def mask_valid(self, value: bool):
        self.p.mask_valid = value

    @property
    def priority(self) -> llama_cpp.ggml_sched_priority:
        """Scheduling prio : (0 - normal, 1 - medium, 2 - high, 3 - realtime)."""
        return self.p.priority

    @priority.setter
    def priority(self, value: llama_cpp.ggml_sched_priority):
        self.p.priority = value

    @property
    def strict_cpu(self) -> bool:
        """Use strict CPU placement."""
        return self.p.strict_cpu

    @strict_cpu.setter
    def strict_cpu(self, bint value):
        self.p.strict_cpu = value

    @property
    def poll(self) -> uint32_t:
        """Polling (busywait) level (0 - no polling, 100 - mostly polling)"""
        return self.p.poll

    @poll.setter
    def poll(self, uint32_t value):
        self.p.poll = value


cdef class CommonParamsSpeculative:
    cdef llama_cpp.common_params_speculative *p
    cdef object owner

    @staticmethod
    cdef CommonParamsSpeculative from_ptr(llama_cpp.common_params_speculative *params, object owner):
        cdef CommonParamsSpeculative wrapper = CommonParamsSpeculative.__new__(CommonParamsSpeculative)
        wrapper.p = params
        wrapper.owner = owner
        return wrapper

    def __init__(self):
        raise Exception(f"Can't construct an instance of {type(self).__name__}")

    @property
    def n_ctx(self) -> int:
        """draft context size."""
        return self.p.n_ctx

    @n_ctx.setter
    def n_ctx(self, value: int):
        self.p.n_ctx = value

    @property
    def n_max(self) -> int:
        """maximum number of tokens to draft during speculative decoding."""
        return self.p.n_max

    @n_max.setter
    def n_max(self, value: int):
        self.p.n_max = value

    @property
    def n_min(self) -> int:
        """minimum number of draft tokens to use for speculative decoding."""
        return self.p.n_min

    @n_min.setter
    def n_min(self, value: int):
        self.p.n_min = value

    @property
    def n_gpu_layers(self) -> int:
        """number of layers to store in VRAM (-1 - use default)."""
        return self.p.n_gpu_layers

    @n_gpu_layers.setter
    def n_gpu_layers(self, value: int):
        self.p.n_gpu_layers = value

    @property
    def p_split(self) -> float:
        """speculative decoding split probability."""
        return self.p.p_split

    @p_split.setter
    def p_split(self, value: float):
        self.p.p_split = value

    @property
    def p_min(self) -> float:
        """minimum speculative decoding probability (greedy)."""
        return self.p.p_min

    @p_min.setter
    def p_min(self, value: float):
        self.p.p_min = value

    @property
    def cpuparams(self) -> CpuParams:
        return CpuParams.from_ptr(&self.p.cpuparams, self)

    @cpuparams.setter
    def cpuparams(self, value: CpuParams):
        self.p.cpuparams = deref(value.p)

    @property
    def cpuparams_batch(self) -> CpuParams:
        return CpuParams.from_ptr(&self.p.cpuparams_batch, self)

    @cpuparams_batch.setter
    def cpuparams_batch(self, value: CpuParams):
        self.p.cpuparams_batch = deref(value.p)

    @property
    def model(self) -> str:
        """ draft model for speculative decoding."""
        return self.p.model

    @model.setter
    def model(self, value: str):
        self.p.model = value


cdef class CommonParamsVocoder:
    cdef llama_cpp.common_params_vocoder *p
    cdef object owner

    @staticmethod
    cdef CommonParamsVocoder from_ptr(llama_cpp.common_params_vocoder *params, owner):
        cdef CommonParamsVocoder wrapper = CommonParamsVocoder.__new__(CommonParamsVocoder)
        wrapper.p = params
        wrapper.owner = owner
        return wrapper

    def __init__(self):
        raise Exception(f"Can't construct an instance of {type(self).__name__}")

    @property
    def hf_repo(self) -> str:
        """HF repo"""
        return self.p.hf_repo.decode()

    @hf_repo.setter
    def hf_repo(self, value: str):
        self.p.hf_repo = value.encode()

    @property
    def hf_file(self) -> str:
        """HF file"""
        return self.p.hf_file.decode()

    @hf_file.setter
    def hf_file(self, value: str):
        self.p.hf_file = value.encode()

    @property
    def model(self) -> str:
        """model path"""
        return self.p.model.decode()

    @model.setter
    def model(self, value: str):
        self.p.model = value.encode()

    @property
    def model_url(self) -> str:
        """model url to download."""
        return self.p.model_url.decode()

    @model_url.setter
    def model_url(self, value: str):
        self.p.model_url = value.encode()


cdef class CommonParams:
    cdef llama_cpp.common_params p

    @property
    def n_predict(self) -> int:
        """new tokens to predict."""
        return self.p.n_predict

    @n_predict.setter
    def n_predict(self, value: int):
        self.p.n_predict = value

    @property
    def n_ctx(self) -> int:
        """context size."""
        return self.p.n_ctx

    @n_ctx.setter
    def n_ctx(self, value: int):
        self.p.n_ctx = value

    @property
    def n_batch(self) -> int:
        """logical batch size for prompt processing (must be >=32)."""
        return self.p.n_batch

    @n_batch.setter
    def n_batch(self, value: int):
        self.p.n_batch = value

    @property
    def n_ubatch(self) -> int:
        """physical batch size for prompt processing (must be >=32)."""
        return self.p.n_ubatch

    @n_ubatch.setter
    def n_ubatch(self, value: int):
        self.p.n_ubatch = value

    @property
    def n_keep(self) -> int:
        """number of tokens to keep from initial prompt."""
        return self.p.n_keep

    @n_keep.setter
    def n_keep(self, value: int):
        self.p.n_keep = value

    @property
    def n_chunks(self) -> int:
        """max number of chunks to process (-1 = unlimited)."""
        return self.p.n_chunks

    @n_chunks.setter
    def n_chunks(self, value: int):
        self.p.n_chunks = value

    @property
    def n_parallel(self) -> int:
        """number of parallel sequences to decode."""
        return self.p.n_parallel

    @n_parallel.setter
    def n_parallel(self, value: int):
        self.p.n_parallel = value

    @property
    def n_sequences(self) -> int:
        """number of sequences to decode."""
        return self.p.n_sequences

    @n_sequences.setter
    def n_sequences(self, value: int):
        self.p.n_sequences = value

    @property
    def grp_attn_n(self) -> int:
        """group-attention factor."""
        return self.p.grp_attn_n

    @grp_attn_n.setter
    def grp_attn_n(self, value: int):
        self.p.grp_attn_n = value

    @property
    def grp_attn_w(self) -> int:
        """group-attention width."""
        return self.p.grp_attn_w

    @grp_attn_w.setter
    def grp_attn_w(self, value: int):
        self.p.grp_attn_w = value

    @property
    def n_print(self) -> int:
        """print token count every n tokens (-1 = disabled)."""
        return self.p.n_print

    @n_print.setter
    def n_print(self, value: int):
        self.p.n_print = value

    @property
    def rope_freq_base(self) -> float:
        """RoPE base frequency."""
        return self.p.rope_freq_base

    @rope_freq_base.setter
    def rope_freq_base(self, value: float):
        self.p.rope_freq_base = value

    @property
    def rope_freq_scale(self) -> float:
        """RoPE frequency scaling factor."""
        return self.p.rope_freq_scale

    @rope_freq_scale.setter
    def rope_freq_scale(self, value: float):
        self.p.rope_freq_scale = value

    @property
    def yarn_ext_factor(self) -> float:
        """YaRN extrapolation mix factor."""
        return self.p.yarn_ext_factor

    @yarn_ext_factor.setter
    def yarn_ext_factor(self, value: float):
        self.p.yarn_ext_factor = value

    @property
    def yarn_attn_factor(self) -> float:
        """YaRN magnitude scaling factor."""
        return self.p.yarn_attn_factor

    @yarn_attn_factor.setter
    def yarn_attn_factor(self, value: float):
        self.p.yarn_attn_factor = value

    @property
    def yarn_beta_fast(self) -> float:
        """YaRN low correction dim."""
        return self.p.yarn_beta_fast

    @yarn_beta_fast.setter
    def yarn_beta_fast(self, value: float):
        self.p.yarn_beta_fast = value

    @property
    def yarn_beta_slow(self) -> float:
        """YaRN high correction dim."""
        return self.p.yarn_beta_slow

    @yarn_beta_slow.setter
    def yarn_beta_slow(self, value: float):
        self.p.yarn_beta_slow = value


    @property
    def yarn_orig_ctx(self) -> int:
        """YaRN original context length."""
        return self.p.yarn_orig_ctx

    @yarn_orig_ctx.setter
    def yarn_orig_ctx(self, value: int):
        self.p.yarn_orig_ctx = value

    @property
    def defrag_thold(self) -> float:
        """KV cache defragmentation threshold."""
        return self.p.defrag_thold

    @defrag_thold.setter
    def defrag_thold(self, value: float):
        self.p.defrag_thold = value

    @property
    def n_gpu_layers(self) -> int:
        """number of layers to store in VRAM (-1 - use default)."""
        return self.p.n_gpu_layers

    @n_gpu_layers.setter
    def n_gpu_layers(self, value: int):
        self.p.n_gpu_layers = value

    @property
    def main_gpu(self) -> int:
        """he GPU that is used for scratch and small tensors"""
        return self.p.main_gpu

    @main_gpu.setter
    def main_gpu(self, value: int):
        self.p.main_gpu = value

    @property
    def tensor_split(self) -> list[float]:
        """how split tensors should be distributed across GPUs."""
        result = []
        for i in range(128):
            result.append(self.p.tensor_split[i])
        return result

    @tensor_split.setter
    def tensor_split(self, value: list[float]):
        assert len(value) == 128, "tensor must of length 128"
        for i in range(128):
            self.p.tensor_split[i] = value[i]

    @property
    def split_mode(self) -> llama_split_mode:
        """how to split the model across GPUs."""
        return self.p.split_mode

    @split_mode.setter
    def split_mode(self, llama_split_mode value):
        self.p.split_mode = value

    @property
    def cpuparams(self) -> CpuParams:
        return CpuParams.from_ptr(&self.p.cpuparams, self)

    @cpuparams.setter
    def cpuparams(self, value: CpuParams):
        self.p.cpuparams = deref(value.p)

    @property
    def cpuparams_batch(self) -> CpuParams:
        return CpuParams.from_ptr(&self.p.cpuparams_batch, self)

    @cpuparams_batch.setter
    def cpuparams_batch(self, value: CpuParams):
        self.p.cpuparams_batch = deref(value.p)

    # @property
    # def cb_eval(self) -> py_sched_eval_callback:
    #     """get/set python ggml backend sched eval callback."""
    #     return <object>self.p.cb_eval_user_data

    # @cb_eval.setter
    # def cb_eval(self, object py_sched_eval_callback):
    #     self.p.cb_eval_user_data = <void*>py_sched_eval_callback

    @property
    def numa(self) -> ggml_numa_strategy:
        """KV cache defragmentation threshold."""
        return self.p.numa

    @numa.setter
    def numa(self, value: ggml_numa_strategy):
        self.p.numa = value

    @property
    def rope_scaling_type(self) -> llama_rope_scaling_type:
        """rope scaling type."""
        return llama_rope_scaling_type(self.p.rope_scaling_type)

    @rope_scaling_type.setter
    def rope_scaling_type(self, llama_rope_scaling_type value):
        self.p.rope_scaling_type = value

    @property
    def pooling_type(self) -> llama_pooling_type:
        """pooling type for embeddings."""
        return (self.p.pooling_type)

    @pooling_type.setter
    def pooling_type(self, llama_pooling_type value):
        self.p.pooling_type = value

    @property
    def attention_type(self) -> llama_attention_type:
        """attention type for embeddings."""
        return llama_attention_type(self.p.attention_type)

    @attention_type.setter
    def attention_type(self, llama_attention_type value):
        self.p.attention_type = value

    @property
    def sampling(self) -> CommonParamsSampling:
        """common params sampling."""
        return CommonParamsSampling.from_ptr(&self.p.sampling, self)

    @sampling.setter
    def sampling(self, value: CommonParamsSampling):
        self.p.sampling = deref(value.p)

    @property
    def speculative(self) -> CommonParamsSpeculative:
        """common params speculative."""
        return CommonParamsSpeculative.from_ptr(&self.p.speculative, self)

    @speculative.setter
    def speculative(self, value: CommonParamsSpeculative):
        self.p.speculative = deref(value.p)

    @property
    def vocoder(self) -> CommonParamsVocoder:
        """common params vocoder."""
        return CommonParamsVocoder.from_ptr(&self.p.vocoder, self)

    @vocoder.setter
    def vocoder(self, value: CommonParamsVocoder):
        self.p.vocoder = deref(value.p)

    @property
    def model(self) -> str:
        """model path"""
        return self.p.model.decode()

    @model.setter
    def model(self, value: str):
        self.p.model = value.encode('utf8')

    @property
    def model_alias(self) -> str:
        """model alias"""
        return self.p.model_alias.decode()

    @model_alias.setter
    def model_alias(self, value: str):
        self.p.model_alias = value.encode('utf8')

    @property
    def model_url(self) -> str:
        """model url to download """
        return self.p.model_url.decode()

    @model_url.setter
    def model_url(self, value: str):
        self.p.model_url = value.encode('utf8')

    @property
    def hf_token(self) -> str:
        """hf token"""
        return self.p.hf_token.decode()

    @hf_token.setter
    def hf_token(self, value: str):
        self.p.hf_token = value.encode('utf8')

    @property
    def hf_repo(self) -> str:
        """hf repo"""
        return self.p.hf_repo.decode()

    @hf_repo.setter
    def hf_repo(self, value: str):
        self.p.hf_repo = value.encode('utf8')

    @property
    def hf_file(self) -> str:
        """hf file"""
        return self.p.hf_file.decode()

    @hf_file.setter
    def hf_file(self, value: str):
        self.p.hf_file = value.encode('utf8')

    @property
    def prompt(self) -> str:
        """the prompt text"""
        return self.p.prompt.decode()

    @prompt.setter
    def prompt(self, value: str):
        self.p.prompt = value.encode('utf8')

    @property
    def prompt_file(self) -> str:
        """store the external prompt file name"""
        return self.p.prompt_file.decode()

    @prompt_file.setter
    def prompt_file(self, value: str):
        self.p.prompt_file = value.encode('utf8')

    @property
    def path_prompt_cache(self) -> str:
        """path to file for saving/loading prompt eval state"""
        return self.p.path_prompt_cache.decode()

    @path_prompt_cache.setter
    def path_prompt_cache(self, value: str):
        self.p.path_prompt_cache = value.encode('utf8')

    @property
    def input_prefix(self) -> str:
        """string to prefix user inputs with"""
        return self.p.input_prefix.decode()

    @input_prefix.setter
    def input_prefix(self, value: str):
        self.p.input_prefix = value.encode('utf8')

    @property
    def input_suffix(self) -> str:
        """string to suffix user inputs with"""
        return self.p.input_suffix.decode()

    @input_suffix.setter
    def input_suffix(self, value: str):
        self.p.input_suffix = value.encode('utf8')

    @property
    def lookup_cache_static(self) -> str:
        """path of static ngram cache file for lookup decoding"""
        return self.p.lookup_cache_static.decode()

    @lookup_cache_static.setter
    def lookup_cache_static(self, value: str):
        self.p.lookup_cache_static = value.encode('utf8')

    @property
    def lookup_cache_dynamic(self) -> str:
        """path of dynamic ngram cache file for lookup decoding"""
        return self.p.lookup_cache_dynamic.decode()

    @lookup_cache_dynamic.setter
    def lookup_cache_dynamic(self, value: str):
        self.p.lookup_cache_dynamic = value.encode('utf8')

    @property
    def logits_file(self) -> str:
        """file for saving *all* logits"""
        return self.p.logits_file.decode()

    @logits_file.setter
    def logits_file(self, value: str):
        self.p.logits_file = value.encode('utf8')

    @property
    def in_files(self) -> list[str]:
        """all input files."""
        result = []
        for i in range(self.p.in_files.size()):
            result.append(self.p.in_files[i].decode())
        return result

    @in_files.setter
    def in_files(self, files: list[str]):
        self.p.in_files.clear()
        for i in files:
            self.p.in_files.push_back(i.encode('utf8'))

    @property
    def antiprompt(self) -> list[str]:
        """strings upon which more user input is prompted (a.k.a. reverse prompts)."""
        result = []
        for i in range(self.p.antiprompt.size()):
            result.append(self.p.antiprompt[i].decode())
        return result

    @antiprompt.setter
    def antiprompt(self, values: list[str]):
        self.p.antiprompt.clear()
        for i in values:
            self.p.antiprompt.push_back(i.encode('utf8'))

    # std::vector<llama_model_kv_override> kv_overrides;

    @property
    def lora_init_without_apply(self) -> bool:
        """only load lora to memory, but do not apply it to ctx (user can manually apply lora later using llama_lora_adapter_apply)."""
        return self.p.lora_init_without_apply

    @lora_init_without_apply.setter
    def lora_init_without_apply(self, value: bool):
        self.p.lora_init_without_apply = value

    # std::vector<llama_lora_adapter_info> lora_adapters; // lora adapter path with user defined scale

    # std::vector<llama_control_vector_load_info> control_vectors; // control vector with user defined scale


    @property
    def verbosity(self) -> int:
        """verbosity"""
        return self.p.verbosity

    @verbosity.setter
    def verbosity(self, value: int):
        self.p.verbosity = value

    @property
    def control_vector_layer_start(self) -> int:
        """layer range for control vector"""
        return self.p.control_vector_layer_start

    @control_vector_layer_start.setter
    def control_vector_layer_start(self, value: int):
        self.p.control_vector_layer_start = value

    @property
    def control_vector_layer_end(self) -> int:
        """layer range for control vector"""
        return self.p.control_vector_layer_end

    @control_vector_layer_end.setter
    def control_vector_layer_end(self, value: int):
        self.p.control_vector_layer_end = value

    @property
    def ppl_stride(self) -> int:
        """stride for perplexity calculations. If left at 0, the pre-existing approach will be used."""
        return self.p.ppl_stride

    @ppl_stride.setter
    def ppl_stride(self, value: int):
        self.p.ppl_stride = value

    @property
    def ppl_output_type(self) -> int:
        """0 -> ppl output is as usual, = 1 -> ppl output is num_tokens, ppl, one per line 

        (which is more convenient to use for plotting)
        """
        return self.p.ppl_output_type

    @ppl_output_type.setter
    def ppl_output_type(self, value: int):
        self.p.ppl_output_type = value

    @property
    def hellaswag(self) -> bool:
        """compute HellaSwag score over random tasks from datafile supplied in prompt"""
        return self.p.hellaswag

    @hellaswag.setter
    def hellaswag(self, value: bool):
        self.p.hellaswag = value

    @property
    def hellaswag_tasks(self) -> int:
        """number of tasks to use when computing the HellaSwag score"""
        return self.p.hellaswag_tasks

    @hellaswag_tasks.setter
    def hellaswag_tasks(self, value: int):
        self.p.hellaswag_tasks = value

    @property
    def winogrande(self) -> bool:
        """compute Winogrande score over random tasks from datafile supplied in prompt"""
        return self.p.winogrande

    @winogrande.setter
    def winogrande(self, value: bool):
        self.p.winogrande = value

    @property
    def winogrande_tasks(self) -> int:
        """number of tasks to use when computing the Winogrande score. If 0, all tasks will be computed"""
        return self.p.winogrande_tasks

    @winogrande_tasks.setter
    def winogrande_tasks(self, value: int):
        self.p.winogrande_tasks = value

    @property
    def multiple_choice(self) -> bool:
        """compute TruthfulQA score over random tasks from datafile supplied in prompt"""
        return self.p.multiple_choice

    @multiple_choice.setter
    def multiple_choice(self, value: bool):
        self.p.multiple_choice = value

    @property
    def multiple_choice_tasks(self) -> int:
        """number of tasks to use when computing the TruthfulQA score. If 0, all tasks will be computed"""
        return self.p.multiple_choice_tasks

    @multiple_choice_tasks.setter
    def multiple_choice_tasks(self, value: int):
        self.p.multiple_choice_tasks = value

    @property
    def kl_divergence(self) -> bool:
        """compute KL divergence"""
        return self.p.kl_divergence

    @kl_divergence.setter
    def kl_divergence(self, value: bool):
        self.p.kl_divergence = value

    @property
    def usage(self) -> bool:
        """print usage"""
        return self.p.usage

    @usage.setter
    def usage(self, value: bool):
        self.p.usage = value

    @property
    def use_color(self) -> bool:
        """use color to distinguish generations and inputs"""
        return self.p.use_color

    @use_color.setter
    def use_color(self, value: bool):
        self.p.use_color = value

    @property
    def special(self) -> bool:
        """enable special token output"""
        return self.p.special

    @special.setter
    def special(self, value: bool):
        self.p.special = value

    @property
    def interactive(self) -> bool:
        """interactive mode"""
        return self.p.interactive

    @interactive.setter
    def interactive(self, value: bool):
        self.p.interactive = value

    @property
    def prompt_cache_all(self) -> bool:
        """save user input and generations to prompt cache"""
        return self.p.prompt_cache_all

    @prompt_cache_all.setter
    def prompt_cache_all(self, value: bool):
        self.p.prompt_cache_all = value

    @property
    def prompt_cache_ro(self) -> bool:
        """ open the prompt cache read-only and do not update it"""
        return self.p.prompt_cache_ro

    @prompt_cache_ro.setter
    def prompt_cache_ro(self, value: bool):
        self.p.prompt_cache_ro = value

    @property
    def escape(self) -> bool:
        """escape special characters"""
        return self.p.escape

    @escape.setter
    def escape(self, value: bool):
        self.p.escape = value

    @property
    def multiline_input(self) -> bool:
        """reverse the usage of "\""""
        return self.p.multiline_input

    @multiline_input.setter
    def multiline_input(self, value: bool):
        self.p.multiline_input = value

    @property
    def simple_io(self) -> bool:
        """improves compatibility with subprocesses and limited consoles"""
        return self.p.simple_io

    @simple_io.setter
    def simple_io(self, value: bool):
        self.p.simple_io = value

    @property
    def cont_batching(self) -> bool:
        """insert new sequences for decoding on-the-fly"""
        return self.p.cont_batching

    @cont_batching.setter
    def cont_batching(self, value: bool):
        self.p.cont_batching = value

    @property
    def flash_attn(self) -> bool:
        """flash attention"""
        return self.p.flash_attn

    @flash_attn.setter
    def flash_attn(self, value: bool):
        self.p.flash_attn = value

    @property
    def no_perf(self) -> bool:
        """disable performance metrics"""
        return self.p.no_perf

    @no_perf.setter
    def no_perf(self, value: bool):
        self.p.no_perf = value

    @property
    def ctx_shift(self) -> bool:
        """context shift on inifinite text generation"""
        return self.p.ctx_shift

    @ctx_shift.setter
    def ctx_shift(self, value: bool):
        self.p.ctx_shift = value

    @property
    def input_prefix_bos(self) -> bool:
        """prefix BOS to user inputs, preceding input_prefix"""
        return self.p.input_prefix_bos

    @input_prefix_bos.setter
    def input_prefix_bos(self, value: bool):
        self.p.input_prefix_bos = value

    @property
    def logits_all(self) -> bool:
        """return logits for all tokens in the batch"""
        return self.p.logits_all

    @logits_all.setter
    def logits_all(self, value: bool):
        self.p.logits_all = value

    @property
    def use_mmap(self) -> bool:
        """use mmap for faster loads"""
        return self.p.use_mmap

    @use_mmap.setter
    def use_mmap(self, value: bool):
        self.p.use_mmap = value

    @property
    def use_mlock(self) -> bool:
        """use mlock to keep model in memory"""
        return self.p.use_mlock

    @use_mlock.setter
    def use_mlock(self, value: bool):
        self.p.use_mlock = value

    @property
    def verbose_prompt(self) -> bool:
        """print prompt tokens before generation"""
        return self.p.verbose_prompt

    @verbose_prompt.setter
    def verbose_prompt(self, value: bool):
        self.p.verbose_prompt = value

    @property
    def display_prompt(self) -> bool:
        """print prompt before generation"""
        return self.p.display_prompt

    @display_prompt.setter
    def display_prompt(self, value: bool):
        self.p.display_prompt = value

    @property
    def dump_kv_cache(self) -> bool:
        """dump the KV cache contents for debugging purposes"""
        return self.p.dump_kv_cache

    @dump_kv_cache.setter
    def dump_kv_cache(self, value: bool):
        self.p.dump_kv_cache = value

    @property
    def no_kv_offload(self) -> bool:
        """disable KV offloading"""
        return self.p.no_kv_offload

    @no_kv_offload.setter
    def no_kv_offload(self, value: bool):
        self.p.no_kv_offload = value

    @property
    def warmup(self) -> bool:
        """warmup run"""
        return self.p.warmup

    @warmup.setter
    def warmup(self, value: bool):
        self.p.warmup = value

    @property
    def check_tensors(self) -> bool:
        """validate tensor data"""
        return self.p.check_tensors

    @check_tensors.setter
    def check_tensors(self, value: bool):
        self.p.check_tensors = value

    @property
    def cache_type_k(self) -> ggml_type:
        """data type for K cache"""
        return <ggml_type>self.p.cache_type_k

    @cache_type_k.setter
    def cache_type_k(self, ggml_type value):
        self.p.cache_type_k = value

    @property
    def cache_type_v(self) -> ggml_type:
        """data type for V cache"""
        return <ggml_type>self.p.cache_type_v

    @cache_type_v.setter
    def cache_type_v(self, ggml_type value):
        self.p.cache_type_v = value

    @property
    def mmproj(self) -> str:
        """path to multimodal projector"""
        return self.p.mmproj.decode()

    @mmproj.setter
    def mmproj(self, value: str):
        self.p.mmproj = value.encode('utf8')

    @property
    def image(self) -> list[str]:
        """paths to image file(s)"""
        result = []
        for i in range(self.p.image.size()):
            result.append(self.p.image[i].decode())
        return result

    @image.setter
    def image(self, files: list[str]):
        self.p.image.clear()
        for i in files:
            self.p.image.push_back(i.encode('utf8'))

    @property
    def embedding(self) -> bool:
        """get only sentence embedding"""
        return self.p.embedding

    @embedding.setter
    def embedding(self, value: bool):
        self.p.embedding = value

    @property
    def embd_normalize(self) -> int:
        """normalisation for embendings (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)"""
        return self.p.embd_normalize

    @embd_normalize.setter
    def embd_normalize(self, value: int):
        self.p.embd_normalize = value

    @property
    def embd_out(self) -> str:
        """empty = default, "array" = [[],[]...], "json" = openai style, "json+" = same "json" + cosine similarity matrix"""
        return self.p.embd_out.decode()

    @embd_out.setter
    def embd_out(self, value: str):
        self.p.embd_out = value.encode('utf8')

    @property
    def embd_sep(self) -> str:
        """separator of embendings"""
        return self.p.embd_sep.decode()

    @embd_sep.setter
    def embd_sep(self, value: str):
        self.p.embd_sep = value.encode('utf8')

    @property
    def reranking(self) -> bool:
        """enable reranking support on server"""
        return self.p.reranking

    @reranking.setter
    def reranking(self, value: bool):
        self.p.reranking = value

    @property
    def port(self) -> int:
        """server listens on this network port"""
        return self.p.port

    @port.setter
    def port(self, value: int):
        self.p.port = value

    @property
    def timeout_read(self) -> int:
        """http read timeout in seconds"""
        return self.p.timeout_read

    @timeout_read.setter
    def timeout_read(self, value: int):
        self.p.timeout_read = value

    @property
    def timeout_write(self) -> int:
        """http write timeout in seconds"""
        return self.p.timeout_write

    @timeout_write.setter
    def timeout_write(self, value: int):
        self.p.timeout_write = value

    @property
    def n_threads_http(self) -> int:
        """number of threads to process HTTP requests (TODO: support threadpool)"""
        return self.p.n_threads_http

    @n_threads_http.setter
    def n_threads_http(self, value: int):
        self.p.n_threads_http = value

    @property
    def n_cache_reuse(self) -> int:
        """min chunk size to reuse from the cache via KV shifting"""
        return self.p.n_cache_reuse

    @n_cache_reuse.setter
    def n_cache_reuse(self, value: int):
        self.p.n_cache_reuse = value

    @property
    def hostname(self) -> str:
        """server hostname"""
        return self.p.hostname.decode()

    @hostname.setter
    def hostname(self, value: str):
        self.p.hostname = value.encode('utf8')

    @property
    def public_path(self) -> str:
        """server public_path"""
        return self.p.public_path.decode()

    @public_path.setter
    def public_path(self, value: str):
        self.p.public_path = value.encode('utf8')

    @property
    def chat_template(self) -> str:
        """chat template"""
        return self.p.chat_template.decode()

    @chat_template.setter
    def chat_template(self, value: str):
        self.p.chat_template = value.encode('utf8')

    # @property
    # def system_prompt(self) -> str:
    #     """system prompt"""
    #     return self.p.system_prompt.decode()

    # @system_prompt.setter
    # def system_prompt(self, value: str):
    #     self.p.system_prompt = value.encode('utf8')

    @property
    def enable_chat_template(self) -> bool:
        """enable chat template"""
        return self.p.enable_chat_template

    @enable_chat_template.setter
    def enable_chat_template(self, value: bool):
        self.p.enable_chat_template = value

    @property
    def api_keys(self) -> list[str]:
        """list of api keys"""
        result = []
        for i in range(self.p.api_keys.size()):
            result.append(self.p.api_keys[i].decode())
        return result

    @api_keys.setter
    def api_keys(self, files: list[str]):
        self.p.api_keys.clear()
        for i in files:
            self.p.api_keys.push_back(i.encode('utf8'))

    @property
    def ssl_file_key(self) -> str:
        """ssl file key"""
        return self.p.ssl_file_key.decode()

    @ssl_file_key.setter
    def ssl_file_key(self, value: str):
        self.p.ssl_file_key = value.encode('utf8')

    @property
    def ssl_file_cert(self) -> str:
        """ssl file cert"""
        return self.p.ssl_file_cert.decode()

    @ssl_file_cert.setter
    def ssl_file_cert(self, value: str):
        self.p.ssl_file_cert = value.encode('utf8')

    @property
    def webui(self) -> bool:
        """enable webui"""
        return self.p.webui

    @webui.setter
    def webui(self, value: bool):
        self.p.webui = value

    @property
    def endpoint_slots(self) -> bool:
        """endpoint slots"""
        return self.p.endpoint_slots

    @endpoint_slots.setter
    def endpoint_slots(self, value: bool):
        self.p.endpoint_slots = value

    @property
    def endpoint_props(self) -> bool:
        """endpoint props"""
        return self.p.endpoint_props

    @endpoint_props.setter
    def endpoint_props(self, value: bool):
        self.p.endpoint_props = value

    @property
    def endpoint_metrics(self) -> bool:
        """endpoint metrics"""
        return self.p.endpoint_metrics

    @endpoint_metrics.setter
    def endpoint_metrics(self, value: bool):
        self.p.endpoint_metrics = value

    @property
    def log_json(self) -> bool:
        """log json"""
        return self.p.log_json

    @log_json.setter
    def log_json(self, value: bool):
        self.p.log_json = value

    @property
    def slot_save_path(self) -> str:
        """slot save path"""
        return self.p.slot_save_path.decode()

    @slot_save_path.setter
    def slot_save_path(self, value: str):
        self.p.slot_save_path = value.encode('utf8')

    @property
    def slot_prompt_similarity(self) -> float:
        """slot prompt similarity."""
        return self.p.slot_prompt_similarity

    @slot_prompt_similarity.setter
    def slot_prompt_similarity(self, value: float):
        self.p.slot_prompt_similarity = value

    @property
    def is_pp_shared(self) -> bool:
        """batched-bench params"""
        return self.p.is_pp_shared

    @is_pp_shared.setter
    def is_pp_shared(self, value: bool):
        self.p.is_pp_shared = value

    @property
    def n_pp(self) -> list[int]:
        return self.p.n_pp

    @n_pp.setter
    def n_pp(self, list[int] values):
        self.p.n_pp = values

    @property
    def n_tg(self) -> list[int]:
        return self.p.n_tg

    @n_tg.setter
    def n_tg(self, list[int] values):
        self.p.n_tg = values

    @property
    def n_pl(self) -> list[int]:
        return self.p.n_pl

    @n_pl.setter
    def n_pl(self, list[int] values):
        self.p.n_pl = values

    @property
    def context_files(self) -> list[str]:
        """context files to embed"""
        return [name.decode() for name in self.p.context_files]

    @context_files.setter
    def context_files(self, list[str] values):
        self.p.context_files = [name.encode() for name in values]

    @property
    def chunk_size(self) -> int:
        """chunk size for context embedding"""
        return self.p.chunk_size

    @chunk_size.setter
    def chunk_size(self, value: int):
        self.p.chunk_size = value

    @property
    def chunk_separator(self) -> str:
        """chunk separator for context embedding"""
        return self.p.chunk_separator.decode()

    @chunk_separator.setter
    def chunk_separator(self, value: str):
        self.p.chunk_separator = value.encode('utf8')

    @property
    def n_junk(self) -> int:
        """number of times to repeat the junk text"""
        return self.p.n_junk

    @n_junk.setter
    def n_junk(self, value: int):
        self.p.n_junk = value

    @property
    def i_pos(self) -> int:
        """position of the passkey in the junk text"""
        return self.p.i_pos

    @i_pos.setter
    def i_pos(self, value: int):
        self.p.i_pos = value

    @property
    def out_file(self) -> str:
        """save the resulting imatrix to this file"""
        return self.p.out_file.decode()

    @out_file.setter
    def out_file(self, value: str):
        self.p.out_file = value.encode('utf8')

    @property
    def n_out_freq(self) -> int:
        """output the imatrix every n_out_freq iterations"""
        return self.p.n_out_freq

    @n_out_freq.setter
    def n_out_freq(self, value: int):
        self.p.n_out_freq = value

    @property
    def n_save_freq(self) -> int:
        """save the imatrix every n_save_freq iterations"""
        return self.p.n_save_freq

    @n_save_freq.setter
    def n_save_freq(self, value: int):
        self.p.n_save_freq = value

    @property
    def i_chunk(self) -> int:
        """start processing from this chunk"""
        return self.p.i_chunk

    @i_chunk.setter
    def i_chunk(self, value: int):
        self.p.i_chunk = value

    @property
    def process_output(self) -> bool:
        """collect data for the output tensor"""
        return self.p.process_output

    @process_output.setter
    def process_output(self, value: bool):
        self.p.process_output = value

    @property
    def compute_ppl(self) -> bool:
        """whether to compute perplexity"""
        return self.p.compute_ppl

    @compute_ppl.setter
    def compute_ppl(self, value: bool):
        self.p.compute_ppl = value

    @property
    def n_pca_batch(self) -> int:
        """start processing from this chunk"""
        return self.p.n_pca_batch

    @n_pca_batch.setter
    def n_pca_batch(self, value: int):
        self.p.n_pca_batch = value

    @property
    def n_pca_iterations(self) -> int:
        """start processing from this chunk"""
        return self.p.n_pca_iterations

    @n_pca_iterations.setter
    def n_pca_iterations(self, value: int):
        self.p.n_pca_iterations = value

    # // cvector-generator params
    # dimre_method cvector_dimre_method = DIMRE_METHOD_PCA;
    # std::string cvector_outfile       =
    # std::string cvector_positive_file = "examples/cvector-generator/positive.txt";
    # std::string cvector_negative_file = "examples/cvector-generator/negative.txt";

    # bool spm_infill = false; // suffix/prefix/middle pattern for infill

    # std::string lora_outfile = "ggml-lora-merged-f16.gguf";

    # // batched-bench params
    # bool batched_bench_output_jsonl = false;


cdef void callback_wrapper(const string &data, void *py_cb) noexcept nogil:
    with gil:
        (<object>py_cb)(data)


cdef class Server:
    cdef shared_ptr[CServer] svr

    def __cinit__(self, CommonParams common_params):
        self.svr = make_shared[CServer](common_params.p)

    def handle_completions(self, string prompt_json_str, res_error, res_ok):
        with nogil:
            self.svr.get().handle_completions(prompt_json_str, callback_wrapper, <void*>res_error, callback_wrapper, <void*>res_ok)

    def handle_chat_completions(self, string prompt_json_str, res_error, res_ok):
        with nogil:
            self.svr.get().handle_chat_completions(prompt_json_str, callback_wrapper, <void*>res_error, callback_wrapper, <void*>res_ok)

    def handle_metrics(self, res_error, res_ok):
        with nogil:
            self.svr.get().handle_metrics(callback_wrapper, <void*>res_error, callback_wrapper, <void*>res_ok)
