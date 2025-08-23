# distutils: language=c++

from xllamacpp.xllamacpp cimport common_params, ggml_backend_dev_props, llama_model_tensor_buft_override
from libcpp.string cimport string as std_string
from libcpp.vector cimport vector as std_vector

cdef extern from "server.h" namespace "xllamacpp" nogil:
    std_string c_get_system_info "xllamacpp::get_system_info" ()

    std_vector[ggml_backend_dev_props] c_get_device_info "xllamacpp::get_device_info" ()

    ctypedef void (*Callback "xllamacpp::Callback")(const std_string &, void *py_cb)
    cdef cppclass CServer "xllamacpp::Server":

        CServer(const common_params& params) except +

        std_string handle_metrics() except +
        
        std_string handle_completions(const std_string &prompt_json_str) except +
        
        std_string handle_chat_completions(const std_string &prompt_json_str) except +

        std_string handle_embeddings(const std_string &input_json_str) except +

    void c_parse_tensor_buffer_overrides "xllamacpp::parse_tensor_buffer_overrides" (
        const std_string & value, std_vector[llama_model_tensor_buft_override] & overrides) except +
    void c_build_tensor_buffer_overrides "xllamacpp::build_tensor_buffer_overrides" (
        const std_vector[llama_model_tensor_buft_override] & overrides, std_string & value) except +
