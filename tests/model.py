import sys
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / 'src'))

import cyllama.cyllama as cy

model_path = str(ROOT / 'models' / 'Llama-3.2-1B-Instruct-Q8_0.gguf')


def progress_callback(progress: float) -> bool:
    return progress > 0.50

cy.llama_backend_init()

params = cy.ModelParams()

params.use_mmap = False
# params.progress_callback = progress_callback

model = cy.LlamaModel(model_path, params)
# model = cy.LlamaModel(model_path)
# assert isinstance(model, cy.LlamaModel)

cy.llama_backend_free()
