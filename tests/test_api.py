import sys
from pathlib import Path
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'src'))

from cyllama import Llama



def test_api(model_path):
    llm = Llama(model_path=model_path, disable_log=True, n_predict=32, n_ctx=512)
    prompt = "When did the universe begin?"
    assert llm.ask(prompt)


