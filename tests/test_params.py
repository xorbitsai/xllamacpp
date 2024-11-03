import platform
import sys
from pathlib import Path

PLATFORM = platform.system()
ARCH = platform.machine()
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'src'))

import cyllama.cyllama as cy


def test_default_model_params():
    params = cy.ModelParams()
    if (PLATFORM, ARCH) == ("Darwin", "arm64"): # i.e. GGML_USE_METAL=ON
        assert params.n_gpu_layers == 999
    else:
        assert params.n_gpu_layers == 0
    assert params.split_mode == cy.LLAMA_SPLIT_MODE_LAYER
    assert params.main_gpu == 0
    assert params.vocab_only == False
    assert params.use_mmap == True
    assert params.use_mlock == False
    assert params.check_tensors == False

def test_default_context_params():
    params = cy.ContextParams()
    assert params.n_ctx               == 512
    assert params.n_batch             == 2048
    assert params.n_ubatch            == 512
    assert params.n_seq_max           == 1
    assert params.n_threads           == cy.GGML_DEFAULT_N_THREADS
    assert params.n_threads_batch     == cy.GGML_DEFAULT_N_THREADS
    assert params.rope_scaling_type   == cy.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED
    assert params.pooling_type        == cy.LLAMA_POOLING_TYPE_UNSPECIFIED
    assert params.attention_type      == cy.LLAMA_ATTENTION_TYPE_UNSPECIFIED
    assert params.rope_freq_base      == 0.0
    assert params.rope_freq_scale     == 0.0
    assert params.yarn_ext_factor     == -1.0
    assert params.yarn_attn_factor    == 1.0
    assert params.yarn_beta_fast      == 32.0
    assert params.yarn_beta_slow      == 1.0
    assert params.yarn_orig_ctx       == 0
    assert params.defrag_thold        == -1.0
    # assert params.cb_eval             == nullptr
    # assert params.cb_eval_user_data   == nullptr
    assert params.type_k              == cy.GGML_TYPE_F16
    assert params.type_v              == cy.GGML_TYPE_F16
    assert params.logits_all          == False
    assert params.embeddings          == False
    assert params.offload_kqv         == True
    assert params.flash_attn          == False
    assert params.no_perf             == True
    # assert params.abort_callback      == nullptr
    # assert params.abort_callback_data == nullptr
