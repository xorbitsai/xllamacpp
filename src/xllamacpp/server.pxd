# distutils: language=c++

from llama_cpp cimport common_params
from libcpp.string cimport string as std_string
from libcpp.functional cimport function as std_function

cdef extern from "server.h" namespace "xllamacpp" nogil:
    ctypedef void (*Callback "xllamacpp::Callback")(const std_string &, void *py_cb)
    cdef cppclass CServer "xllamacpp::Server":

        CServer(const common_params& params) except +
        
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
