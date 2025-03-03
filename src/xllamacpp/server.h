#include "common.h"
#include <memory>
#include <string>
#include <thread>

struct server_context;

namespace xllamacpp {
typedef void (*Callback)(const std::string &, void *py_cb);

class Server {
public:
  Server(const common_params &params);
  ~Server();

  void handle_completions(const std::string &prompt_json_str,
                          Callback res_error, void *py_cb_error,
                          Callback res_ok, void *py_cb_ok);

private:
  common_params _params;
  std::shared_ptr<server_context>
      _ctx_server; // incomplete type of server_context
  std::thread _loop_thread;
};
} // namespace xllamacpp
