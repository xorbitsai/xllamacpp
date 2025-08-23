#include "common.h"
#include <memory>
#include <string>
#include <thread>
#include <vector>

struct server_context;

namespace xllamacpp {

std::string get_system_info();

std::vector<ggml_backend_dev_props> get_device_info();

typedef void (*Callback)(const std::string &, void *py_cb);

class Server {
public:
  Server(const common_params &params);
  ~Server();

  std::string handle_metrics();

  std::string handle_completions(const std::string &prompt_json_str);

  std::string handle_chat_completions(const std::string &prompt_json_str);

  std::string handle_embeddings(const std::string &input_json_str);

private:
  common_params _params;
  std::shared_ptr<server_context>
      _ctx_server; // incomplete type of server_context
  std::thread _loop_thread;
};
} // namespace xllamacpp
