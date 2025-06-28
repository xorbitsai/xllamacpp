# distutils: language = c++
# cython: profile=False
# cython: embedsignature = True
# cython: language_level = 3
# cython: c_string_encoding = default

"""
xllamacpp: a thin cython wrapper of llama.cpp
"""
from libc.stdint cimport int32_t, uint32_t
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr, make_shared
from cython.operator cimport dereference as deref

cimport xllamacpp
from server cimport CServer, c_get_device_info


# constants
# -----------------------------------------------------------------------------

LLAMA_DEFAULT_SEED = 0xFFFFFFFF


# build info
# -----------------------------------------------------------------------------

BUILD_INFO = {
    'build_number': xllamacpp.LLAMA_BUILD_NUMBER,
    'commit': xllamacpp.LLAMA_COMMIT.decode(),
    'compiler': xllamacpp.LLAMA_COMPILER.decode(),
    'build_target': xllamacpp.LLAMA_BUILD_TARGET.decode(),
}


cdef class LlamaLogitBias:
    cdef xllamacpp.llama_logit_bias *p
    cdef object owner

    @staticmethod
    cdef LlamaLogitBias from_ptr(xllamacpp.llama_logit_bias *p, object owner):
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
    cdef xllamacpp.common_params_sampling *p
    cdef object owner

    @staticmethod
    cdef CommonParamsSampling from_ptr(xllamacpp.common_params_sampling *params, object owner):
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
    def samplers(self) -> str:
        """get/set sampler types
        
        std_vector[common_sampler_type] samplers
        """
        res = []
        for sampler_enum in self.p.samplers:
            res.append(<str>xllamacpp.common_sampler_type_to_str(sampler_enum))
        return ";".join(res)

    @samplers.setter
    def samplers(self, value: str):
        cdef vector[string] split_values = value.split(";")
        self.p.samplers = xllamacpp.common_sampler_types_from_names(split_values, True)

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
        cdef vector[xllamacpp.llama_logit_bias] vec
        for elem in elems:
            vec.push_back(elem.ptr[0])
        self.p.logit_bias = vec



cdef class CpuParams:
    cdef xllamacpp.cpu_params *p
    cdef object owner

    @staticmethod
    cdef CpuParams from_ptr(xllamacpp.cpu_params *params, object owner):
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
    def priority(self) -> xllamacpp.ggml_sched_priority:
        """Scheduling prio : (0 - normal, 1 - medium, 2 - high, 3 - realtime)."""
        return self.p.priority

    @priority.setter
    def priority(self, value: xllamacpp.ggml_sched_priority):
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


cdef class CommonParamsModel:
    cdef xllamacpp.common_params_model *p
    cdef object owner

    @staticmethod
    cdef CommonParamsModel from_ptr(xllamacpp.common_params_model *params, object owner):
        cdef CommonParamsModel wrapper = CommonParamsModel.__new__(CommonParamsModel)
        wrapper.p = params
        wrapper.owner = owner
        return wrapper

    def __init__(self):
        raise Exception(f"Can't construct an instance of {type(self).__name__}")

    @property
    def path(self) -> str:
        """model local path"""
        return self.p.path.decode()

    @path.setter
    def path(self, value: str):
        self.p.path = value.encode()

    @property
    def url(self) -> str:
        """model url to download"""
        return self.p.url.decode()

    @url.setter
    def url(self, value: str):
        self.p.url = value.encode()

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


cdef class CommonParamsSpeculative:
    cdef xllamacpp.common_params_speculative *p
    cdef object owner

    @staticmethod
    cdef CommonParamsSpeculative from_ptr(xllamacpp.common_params_speculative *params, object owner):
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
    def cache_type_k(self) -> ggml_type:
        """data type for K cache"""
        return self.p.cache_type_k

    @cache_type_k.setter
    def cache_type_k(self, value: ggml_type):
        self.p.cache_type_k = value

    @property
    def cache_type_v(self) -> ggml_type:
        """data type for V cache"""
        return self.p.cache_type_v

    @cache_type_v.setter
    def cache_type_v(self, ggml_type value):
        self.p.cache_type_v = value

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
    def model(self) -> CommonParamsModel:
        return CommonParamsModel.from_ptr(&self.p.model, self)

    @model.setter
    def model(self, value: CommonParamsModel):
        self.p.model = deref(value.p)


cdef class CommonParamsVocoder:
    cdef xllamacpp.common_params_vocoder *p
    cdef object owner

    @staticmethod
    cdef CommonParamsVocoder from_ptr(xllamacpp.common_params_vocoder *params, owner):
        cdef CommonParamsVocoder wrapper = CommonParamsVocoder.__new__(CommonParamsVocoder)
        wrapper.p = params
        wrapper.owner = owner
        return wrapper

    def __init__(self):
        raise Exception(f"Can't construct an instance of {type(self).__name__}")

    @property
    def model(self) -> CommonParamsModel:
        return CommonParamsModel.from_ptr(&self.p.model, self)

    @model.setter
    def model(self, value: CommonParamsModel):
        self.p.model = deref(value.p)

    @property
    def speaker_file(self) -> str:
        """speaker file path"""
        return self.p.speaker_file.decode()

    @speaker_file.setter
    def speaker_file(self, value: str):
        self.p.speaker_file = value.encode()


cdef class CommonParams:
    cdef xllamacpp.common_params p

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
        return self.p.rope_scaling_type

    @rope_scaling_type.setter
    def rope_scaling_type(self, llama_rope_scaling_type value):
        self.p.rope_scaling_type = value

    @property
    def pooling_type(self) -> llama_pooling_type:
        """pooling type for embeddings."""
        return self.p.pooling_type

    @pooling_type.setter
    def pooling_type(self, llama_pooling_type value):
        self.p.pooling_type = value

    @property
    def attention_type(self) -> llama_attention_type:
        """attention type for embeddings."""
        return self.p.attention_type

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
    def model(self) -> CommonParamsModel:
        return CommonParamsModel.from_ptr(&self.p.model, self)

    @model.setter
    def model(self, value: CommonParamsModel):
        self.p.model = deref(value.p)

    @property
    def model_alias(self) -> str:
        """model alias"""
        return self.p.model_alias.decode()

    @model_alias.setter
    def model_alias(self, value: str):
        self.p.model_alias = value.encode('utf8')

    @property
    def hf_token(self) -> str:
        """hf token"""
        return self.p.hf_token.decode()

    @hf_token.setter
    def hf_token(self, value: str):
        self.p.hf_token = value.encode('utf8')

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
    # std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;

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
    def offline(self) -> bool:
        return self.p.offline

    @offline.setter
    def offline(self, value: bool):
        self.p.offline = value

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
    def swa_full(self) -> bool:
        """use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)"""
        return self.p.swa_full

    @swa_full.setter
    def swa_full(self, value: bool):
        self.p.swa_full = value

    @property
    def input_prefix_bos(self) -> bool:
        """prefix BOS to user inputs, preceding input_prefix"""
        return self.p.input_prefix_bos

    @input_prefix_bos.setter
    def input_prefix_bos(self, value: bool):
        self.p.input_prefix_bos = value

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
    def no_op_offload(self) -> bool:
        """globally disable offload host tensor operations to device"""
        return self.p.no_op_offload

    @no_op_offload.setter
    def no_op_offload(self, value: bool):
        self.p.no_op_offload = value

    @property
    def single_turn(self) -> bool:
        """single turn chat conversation"""
        return self.p.single_turn

    @single_turn.setter
    def single_turn(self, value: bool):
        self.p.single_turn = value

    @property
    def cache_type_k(self) -> ggml_type:
        """data type for K cache"""
        return self.p.cache_type_k

    @cache_type_k.setter
    def cache_type_k(self, ggml_type value):
        self.p.cache_type_k = value

    @property
    def cache_type_v(self) -> ggml_type:
        """data type for V cache"""
        return self.p.cache_type_v

    @cache_type_v.setter
    def cache_type_v(self, ggml_type value):
        self.p.cache_type_v = value

    @property
    def mmproj(self) -> CommonParamsModel:
        return CommonParamsModel.from_ptr(&self.p.mmproj, self)

    @mmproj.setter
    def mmproj(self, value: CommonParamsModel):
        self.p.mmproj = deref(value.p)

    @property
    def mmproj_use_gpu(self) -> bool:
        """use GPU for multimodal model"""
        return self.p.mmproj_use_gpu

    @mmproj_use_gpu.setter
    def mmproj_use_gpu(self, value: bool):
        self.p.mmproj_use_gpu = value

    @property
    def no_mmproj(self) -> bool:
        """explicitly disable multimodal model"""
        return self.p.no_mmproj

    @no_mmproj.setter
    def no_mmproj(self, value: bool):
        self.p.no_mmproj = value

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
    def cls_sep(self) -> str:
        """separator of classification sequences"""
        return self.p.cls_sep.decode()

    @cls_sep.setter
    def cls_sep(self, value: str):
        self.p.cls_sep = value.encode('utf8')

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

    @property
    def enable_chat_template(self) -> bool:
        """enable chat template"""
        return self.p.enable_chat_template

    @enable_chat_template.setter
    def enable_chat_template(self, value: bool):
        self.p.enable_chat_template = value

    @property
    def reasoning_format(self) -> common_reasoning_format:
        return self.p.reasoning_format

    @reasoning_format.setter
    def reasoning_format(self, common_reasoning_format value):
        self.p.reasoning_format = value

    @property
    def reasoning_budget(self) -> int:
        return self.p.reasoning_budget

    @reasoning_budget.setter
    def reasoning_budget(self, value: int):
        self.p.reasoning_budget = value

    @property
    def prefill_assistant(self) -> bool:
        """if true, any trailing assistant message will be prefilled into the response"""
        return self.p.prefill_assistant

    @prefill_assistant.setter
    def prefill_assistant(self, value: bool):
        self.p.prefill_assistant = value

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
        """output filename for all example programs"""
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
    def parse_special(self) -> bool:
        """whether to parse special tokens during imatrix tokenization"""
        return self.p.parse_special

    @parse_special.setter
    def parse_special(self, value: bool):
        self.p.parse_special = value

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
    # std::string cvector_positive_file = "tools/cvector-generator/positive.txt";
    # std::string cvector_negative_file = "tools/cvector-generator/negative.txt";

    # bool spm_infill = false; // suffix/prefix/middle pattern for infill

    # std::string lora_outfile = "ggml-lora-merged-f16.gguf";

    # // batched-bench params
    # bool batched_bench_output_jsonl = false;


def get_device_info():
    cdef object data = c_get_device_info()
    for info in data:
        info["name"] = info["name"].decode("utf-8")
        info["description"] = info["description"].decode("utf-8")
    return data


cdef void callback_wrapper(const string &data, void *py_cb) noexcept nogil:
    with gil:
        (<object>py_cb)(data)


cdef class Server:
    cdef shared_ptr[CServer] svr

    def __cinit__(self, CommonParams common_params):
        self.svr = make_shared[CServer](common_params.p)

    def handle_metrics(self, res_error, res_ok):
        with nogil:
            self.svr.get().handle_metrics(callback_wrapper, <void*>res_error, callback_wrapper, <void*>res_ok)

    def handle_completions(self, string prompt_json_str, res_error, res_ok):
        with nogil:
            self.svr.get().handle_completions(prompt_json_str, callback_wrapper, <void*>res_error, callback_wrapper, <void*>res_ok)

    def handle_chat_completions(self, string prompt_json_str, res_error, res_ok):
        with nogil:
            self.svr.get().handle_chat_completions(prompt_json_str, callback_wrapper, <void*>res_error, callback_wrapper, <void*>res_ok)

    def handle_embeddings(self, string input_json_str, res_error, res_ok):
        with nogil:
            self.svr.get().handle_embeddings(input_json_str, callback_wrapper, <void*>res_error, callback_wrapper, <void*>res_ok)
