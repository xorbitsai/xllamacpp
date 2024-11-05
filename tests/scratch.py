import sys
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / 'src'))

import cyllama.cyllama as cy

model_path = str(ROOT / 'models' / 'Llama-3.2-1B-Instruct-Q8_0.gguf')

cy.llama_backend_init()    
model = cy.LlamaModel(model_path)
ctx = cy.LlamaContext(model)
assert ctx.n_ctx > 0
# cy.llama_backend_free()


# params = cy.CommonParams()
# cy.common_init()

# params.model = model_path
# params.prompt = "When did the universe begin?"
# params.n_predict = 32
# params.n_ctx = 512
# params.cpuparams.n_threads = 4

# # total length of the sequence including the prompt
# n_predict: int = params.n_predict

# # init LLM
# cy.llama_backend_init()
# cy.llama_numa_init(params.numa)

# # load the model and apply lora adapter, if any
# llama_init = cy.CommonInitResult(params)
# model = llama_init.model;
# ctx = llama_init.context;

# cy.llama_backend_free()