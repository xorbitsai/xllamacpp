# distutils: language=c++

from xllamacpp.xllamacpp cimport common_params, ggml_backend_dev_props
from libcpp.string cimport string as std_string
from libcpp.vector cimport vector as std_vector

cdef extern from "server.h" namespace "xllamacpp" nogil:
    std_string c_get_system_info "xllamacpp::get_system_info" ()

    std_vector[ggml_backend_dev_props] c_get_device_info "xllamacpp::get_device_info" ()

    ctypedef void (*Callback "xllamacpp::Callback")(const std_string &, void *py_cb)
    cdef cppclass CServer "xllamacpp::Server":

        CServer(const common_params& params) except +

        void handle_metrics(Callback res_error,
                void *py_cb_error,
                Callback res_ok,
                void *py_cb_ok) except +
        
        void handle_completions(const std_string &prompt_json_str,
                Callback res_error,
                void *py_cb_error,
                Callback res_ok,
                void *py_cb_ok) except +
        
        void handle_chat_completions(const std_string &prompt_json_str,
                Callback res_error,
                void *py_cb_error,
                Callback res_ok,
                void *py_cb_ok) except +

        void handle_embeddings(const std_string &input_json_str,
                Callback res_error,
                void *py_cb_error,
                Callback res_ok,
                void *py_cb_ok) except +
