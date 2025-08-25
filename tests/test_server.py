import pprint
import os
import sys
import requests
import base64
import pytest

print(sys.path)
import xllamacpp as xlc


def test_get_system_info():
    assert "CPU :" in xlc.get_system_info()


def test_get_device_info():
    xlc.get_device_info()
    info = xlc.get_device_info()
    assert len(info) > 0
    assert "CPU" in [i["name"] for i in info]


def test_llama_server(model_path):
    params = xlc.CommonParams()

    params.model.path = os.path.join(model_path, "Llama-3.2-1B-Instruct-Q8_0.gguf")
    params.prompt = "When did the universe begin?"
    params.n_predict = 32
    params.n_ctx = 512
    params.cpuparams.n_threads = 4
    params.cpuparams_batch.n_threads = 2
    params.endpoint_metrics = True

    server = xlc.Server(params)

    complete_prompt = {
        "max_tokens": 128,
        "prompt": "Write the fibonacci function in c++.",
    }

    server.handle_completions(
        complete_prompt,
        lambda v: pprint.pprint(v),
    )
    v = server.handle_completions(complete_prompt)
    assert isinstance(v, dict)
    assert "code" not in v
    pprint.pprint(v)

    complete_prompt["stream"] = True

    with pytest.raises(ValueError, match="requires a callback for streaming"):
        server.handle_completions(complete_prompt)

    server.handle_completions(
        complete_prompt,
        lambda v: pprint.pprint(v),
    )

    chat_complete_prompt = {
        "max_tokens": 128,
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Write the fibonacci function in c++."},
        ],
    }

    server.handle_chat_completions(
        chat_complete_prompt,
        lambda v: pprint.pprint(v),
    )
    v = server.handle_chat_completions(chat_complete_prompt)
    assert isinstance(v, dict)
    assert "code" not in v
    pprint.pprint(v)

    chat_complete_prompt["stream"] = True

    with pytest.raises(ValueError, match="requires a callback for streaming"):
        server.handle_chat_completions(chat_complete_prompt)

    server.handle_chat_completions(
        chat_complete_prompt,
        lambda v: pprint.pprint(v),
    )
    result = server.handle_metrics()
    assert type(result) is str
    print(result)


def test_llama_server_multimodal(model_path):
    IMG_URL_0 = (
        "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png"
    )

    response = requests.get(IMG_URL_0)
    response.raise_for_status()  # Raise an exception for bad status codes
    IMG_BASE64_0 = "data:image/png;base64," + base64.b64encode(response.content).decode(
        "utf-8"
    )

    params = xlc.CommonParams()

    params.model.path = os.path.join(model_path, "tinygemma3-Q8_0.gguf")
    params.mmproj.path = os.path.join(model_path, "mmproj-tinygemma3.gguf")
    params.sampling.seed = 42
    params.sampling.top_k = 1
    params.sampling.temp = 0
    params.n_predict = 4
    params.n_ctx = 1024
    params.cpuparams.n_threads = 4
    params.cpuparams_batch.n_threads = 2

    server = xlc.Server(params)

    chat_complete_prompt = {
        "max_tokens": 128,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this:\n"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": IMG_BASE64_0,
                        },
                    },
                ],
            },
        ],
    }

    server.handle_chat_completions(
        chat_complete_prompt,
        lambda v: pprint.pprint(v),
    )


def test_llama_server_embedding(model_path):
    params = xlc.CommonParams()

    params.model.path = os.path.join(model_path, "Qwen3-Embedding-0.6B-Q8_0.gguf")
    params.embedding = True
    params.n_predict = -1
    params.n_ctx = 512
    params.n_batch = 128
    params.n_ubatch = 128
    params.sampling.seed = 42
    params.cpuparams.n_threads = 2
    params.cpuparams_batch.n_threads = 2
    params.pooling_type = xlc.llama_pooling_type.LLAMA_POOLING_TYPE_LAST

    server = xlc.Server(params)

    embedding_input = {
        "input": [
            "I believe the meaning of life is",
            "Write a joke about AI from a very long prompt which will not be truncated",
            "This is a test",
            "This is another test",
        ],
    }

    result = server.handle_embeddings(embedding_input)

    assert type(result) is dict
    assert len(result["data"]) == 4
    for d in result["data"]:
        assert len(d["embedding"]) == 1024
