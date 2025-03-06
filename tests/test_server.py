import json
import pprint
import sys

print(sys.path)
import xllamacpp as xlc


def test_llama_server(model_path):
    params = xlc.CommonParams()

    params.model = model_path
    params.prompt = "When did the universe begin?"
    params.n_predict = 32
    params.n_ctx = 512
    params.cpuparams.n_threads = 4
    params.cpuparams_batch.n_threads = 2

    server = xlc.Server(params)

    complete_prompt = {
        "max_tokens": 128,
        "prompt": "Write the fibonacci function in c++.",
    }

    server.handle_completions(
        json.dumps(complete_prompt),
        lambda s: pprint.pprint(json.loads(s)),
        lambda s: pprint.pprint(json.loads(s)),
    )
    complete_prompt["stream"] = True

    server.handle_completions(
        json.dumps(complete_prompt),
        lambda s: pprint.pprint(json.loads(s)),
        lambda s: pprint.pprint(json.loads(s)),
    )

    chat_complete_prompt = {
        "max_tokens": 128,
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Write the fibonacci function in c++."},
        ],
    }

    server.handle_chat_completions(
        json.dumps(chat_complete_prompt),
        lambda s: pprint.pprint(json.loads(s)),
        lambda s: pprint.pprint(json.loads(s)),
    )

    chat_complete_prompt["stream"] = True

    server.handle_chat_completions(
        json.dumps(chat_complete_prompt),
        lambda s: pprint.pprint(json.loads(s)),
        lambda s: pprint.pprint(json.loads(s)),
    )
