import sys;sys.path.insert(0, 'src')

import pyllama.pyllama as cy
from pyllama import Llama

llm = Llama(model_path='models/Llama-3.2-1B-Instruct-Q8_0.gguf')