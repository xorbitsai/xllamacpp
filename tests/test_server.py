import json
import pprint

import pyllama.pyllama as cy
import os
import time


def test_llama_server(model_path):
    params = cy.CommonParams()

    params.model = model_path
    params.prompt = "When did the universe begin?"
    params.n_predict = 32
    params.n_ctx = 512
    params.cpuparams.n_threads = 4

    server = cy.Server(params)

    prompt = {
        "max_tokens": 128,
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Write the fibonacci function in c++."},
        ],
    }

    server.handle_completions(
        json.dumps(prompt),
        lambda s: pprint.pprint(json.loads(s)),
        lambda s: pprint.pprint(json.loads(s)),
    )

    prompt["stream"] = True
    server.handle_completions(json.dumps(prompt), print, print)
