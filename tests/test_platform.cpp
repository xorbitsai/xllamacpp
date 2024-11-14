#include "llama.h"
#include <iostream>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>


int main() {
    std::string model_path = "models/Llama-3.2-1B-Instruct-Q8_0.gguf";
    std::string prompt = "Is Mathematics invented or discovered?";
    // number of layers to offload to the GPU
    int ngl = 99;
    // number of tokens to predict
    int n_predict = 32;


    // initialize the model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;
    llama_model * model = llama_load_model_from_file(model_path.c_str(), model_params);

    // model properties
    uint64_t n_params = llama_model_n_params(model);
    uint64_t size = llama_model_size(model);

    std::cout << "model.n_params: " << n_params << std::endl;
    std::cout << "model.size: " << size << std::endl;
    std::cout.flush();

    llama_free_model(model);

    return 0;
}
