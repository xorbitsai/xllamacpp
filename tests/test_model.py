
import cyllama.cyllama as cy

def progress_callback(progress: float) -> bool:
    return progress > 0.50

def test_model_instance(model_path):
    cy.llama_backend_init()
    model = cy.LlamaModel(model_path)
    cy.llama_backend_free()

def test_model_load_cancel(model_path):
    cy.llama_backend_init()
    params = cy.LlamaModelParams()
    params.use_mmap = False
    params.progress_callback = progress_callback
    model = cy.LlamaModel(model_path, params)
    cy.llama_backend_free()

def test_autorelease(model_path):
    # need to wrap in a thread here.
    cy.llama_backend_init()    
    model = cy.LlamaModel(model_path)
    ctx = cy.LlamaContext(model)
    cy.llama_backend_free()
