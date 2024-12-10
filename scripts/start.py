import sys;sys.path.insert(0, 'src')

import cyllama.cyllama as cy
from cyllama import Llama

llm = Llama(model_path='models/Llama-3.2-1B-Instruct-Q8_0.gguf')