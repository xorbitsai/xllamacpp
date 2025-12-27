import pprint
import os
import sys
import base64
import pytest
import json
import orjson

import xllamacpp as xlc


def test_get_system_info():
    assert "CPU :" in xlc.get_system_info()


def test_get_device_info():
    xlc.get_device_info()
    info = xlc.get_device_info()
    assert len(info) > 0
    assert "CPU" in [i["name"] for i in info]
    print(info)


def test_llama_server(model_path):
    params = xlc.CommonParams()

    params.model.path = os.path.join(model_path, "Llama-3.2-1B-Instruct-Q8_0.gguf")
    params.prompt = "When did the universe begin?"
    params.warmup = False
    params.n_predict = 32
    params.n_ctx = 256
    params.n_parallel = 1
    params.cpuparams.n_threads = 2
    params.cpuparams_batch.n_threads = 2
    params.endpoint_metrics = True
    params.cache_ram_mib = 0

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

    # If the prompt is a str or bytes, a callback is required.
    with pytest.raises(ValueError, match="non dict prompt"):
        server.handle_chat_completions(orjson.dumps(complete_prompt))

    complete_prompt["stream"] = True

    # If the prompt is streaming, a callback is required.
    with pytest.raises(ValueError, match="requires a callback for streaming"):
        server.handle_completions(complete_prompt)

    server.handle_completions(
        complete_prompt,
        lambda v: pprint.pprint(v),
    )

    # Test handle_completions with a str or bytes prompt
    ok = False

    def _cb_str(v):
        nonlocal ok
        assert type(v) is str
        json.loads(v)
        ok = True

    complete_prompt_str = json.dumps(complete_prompt)
    server.handle_completions(
        complete_prompt_str,
        _cb_str,
    )
    assert ok

    ok = False

    def _cb_bytes(v):
        nonlocal ok
        assert type(v) is bytes
        orjson.loads(v)
        ok = True

    complete_prompt_bytes = orjson.dumps(complete_prompt)
    server.handle_completions(
        complete_prompt_bytes,
        _cb_bytes,
    )
    assert ok

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

    # If the prompt is a str or bytes, a callback is required.
    with pytest.raises(ValueError, match="non dict prompt"):
        server.handle_chat_completions(json.dumps(chat_complete_prompt))

    chat_complete_prompt["stream"] = True

    # If the prompt is streaming, a callback is required.
    with pytest.raises(ValueError, match="requires a callback for streaming"):
        server.handle_chat_completions(chat_complete_prompt)

    server.handle_chat_completions(
        chat_complete_prompt,
        lambda v: pprint.pprint(v),
    )

    # Test handle_chat_completions with a str or bytes prompt
    ok = False

    def _cb_str(v):
        nonlocal ok
        assert type(v) is str
        json.loads(v)
        ok = True

    chat_complete_prompt_str = json.dumps(chat_complete_prompt)
    server.handle_chat_completions(
        chat_complete_prompt_str,
        _cb_str,
    )
    assert ok

    ok = False

    def _cb_bytes(v):
        nonlocal ok
        assert type(v) is bytes
        orjson.loads(v)
        ok = True

    chat_complete_prompt_bytes = orjson.dumps(chat_complete_prompt)
    server.handle_chat_completions(
        chat_complete_prompt_bytes,
        _cb_bytes,
    )
    assert ok

    # Test handle_metrics()
    result = server.handle_metrics()
    assert type(result) is str
    assert "llamacpp:prompt_seconds_total" in result


def test_llama_server_stream_callback_stop(model_path):
    params = xlc.CommonParams()

    params.model.path = os.path.join(model_path, "Llama-3.2-1B-Instruct-Q8_0.gguf")
    params.prompt = "When did the universe begin?"
    params.warmup = False
    params.n_predict = 64
    params.n_ctx = 256
    params.n_parallel = 1
    params.cpuparams.n_threads = 2
    params.cpuparams_batch.n_threads = 2

    server = xlc.Server(params)

    # Test handle_completions streaming stop via callback return value
    complete_prompt = {
        "max_tokens": 128,
        "prompt": "Write a long story about the history of the universe.",
        "stream": True,
    }

    all_chunks = 0

    def _cb_all(v):
        nonlocal all_chunks
        all_chunks += 1

    stop_chunks = 0

    def _cb_stop(v):
        nonlocal stop_chunks
        stop_chunks += 1
        return True

    server.handle_completions(complete_prompt, _cb_all)
    assert all_chunks >= 1

    server.handle_completions(complete_prompt, _cb_stop)
    assert stop_chunks == 1
    assert all_chunks > stop_chunks

    # Test handle_chat_completions streaming stop via callback return value
    chat_complete_prompt = {
        "max_tokens": 128,
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {
                "role": "user",
                "content": "Tell me in detail about the history of programming languages.",
            },
        ],
        "stream": True,
    }

    chat_all_chunks = 0

    def _chat_cb_all(v):
        nonlocal chat_all_chunks
        chat_all_chunks += 1

    chat_stop_chunks = 0

    def _chat_cb_stop(v):
        nonlocal chat_stop_chunks
        chat_stop_chunks += 1
        return True

    server.handle_chat_completions(chat_complete_prompt, _chat_cb_all)
    assert chat_all_chunks >= 1

    server.handle_chat_completions(chat_complete_prompt, _chat_cb_stop)
    assert chat_stop_chunks == 1
    assert chat_all_chunks > chat_stop_chunks


def test_llama_server_chat_with_grammar(model_path):
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "score": {"type": "number"},
        },
        "required": ["answer"],
    }
    grammar = xlc.json_schema_to_grammar(schema)

    params = xlc.CommonParams()

    params.model.path = os.path.join(model_path, "Llama-3.2-1B-Instruct-Q8_0.gguf")
    params.warmup = False
    params.n_predict = 64
    params.n_ctx = 256
    params.cpuparams.n_threads = 2
    params.cpuparams_batch.n_threads = 2
    params.sampling.temp = 0
    params.sampling.top_k = 1
    params.sampling.grammar = grammar

    server = xlc.Server(params)

    chat_complete_prompt = {
        "max_tokens": 64,
        "messages": [
            {
                "role": "system",
                "content": "Respond with a JSON object matching the provided schema.",
            },
            {
                "role": "user",
                "content": "Provide an answer string and an optional numeric score.",
            },
        ],
    }

    result = server.handle_chat_completions(chat_complete_prompt)

    assert isinstance(result, dict)
    content = result["choices"][0]["message"]["content"]
    parsed = json.loads(content)

    assert parsed["answer"]
    assert isinstance(parsed["answer"], str)
    if "score" in parsed:
        assert isinstance(parsed["score"], (int, float))


def test_llama_server_multimodal(model_path):
    with open(os.path.join(os.path.dirname(__file__), "data/11_truck.png"), "rb") as f:
        content = f.read()
    IMG_BASE64_0 = "data:image/png;base64," + base64.b64encode(content).decode("utf-8")

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

    embedding_input_str = json.dumps(embedding_input)
    assert type(embedding_input_str) is str
    result_str = server.handle_embeddings(embedding_input_str)
    assert type(result_str) is str
    result = json.loads(result_str)

    assert type(result) is dict
    assert len(result["data"]) == 4
    for d in result["data"]:
        assert len(d["embedding"]) == 1024

    embedding_input_bytes = orjson.dumps(embedding_input)
    assert type(embedding_input_bytes) is bytes
    result_bytes = server.handle_embeddings(embedding_input_bytes)
    assert type(result_bytes) is bytes
    result = orjson.loads(result_str)

    assert type(result) is dict
    assert len(result["data"]) == 4
    for d in result["data"]:
        assert len(d["embedding"]) == 1024


@pytest.mark.skipif(sys.platform == "darwin", reason="Rerank test crashes on macOS CI")
def test_llama_server_rerank(model_path):
    params = xlc.CommonParams()

    params.model.path = os.path.join(model_path, "bge-reranker-v2-m3-Q2_K.gguf")
    params.embedding = True
    params.n_predict = -1
    params.n_ctx = 512
    params.n_batch = 128
    params.n_ubatch = 128
    params.sampling.seed = 42
    params.cpuparams.n_threads = 2
    params.cpuparams_batch.n_threads = 2
    params.pooling_type = xlc.llama_pooling_type.LLAMA_POOLING_TYPE_RANK

    server = xlc.Server(params)

    rerank_input = {
        "query": "What is the capital of France?",
        "documents": [
            "Paris is the capital of France.",
            "The Eiffel Tower is in Paris.",
            "Germany is located in Europe.",
        ],
    }

    result = server.handle_rerank(rerank_input)

    assert type(result) is dict
    assert len(result["results"]) == 3

    rerank_input_str = json.dumps(rerank_input)
    result_str = server.handle_rerank(rerank_input_str)
    assert type(result_str) is str
    result = json.loads(result_str)

    assert type(result) is dict
    assert len(result["results"]) == 3

    rerank_input_bytes = orjson.dumps(rerank_input)
    result_bytes = server.handle_rerank(rerank_input_bytes)
    assert type(result_bytes) is bytes
    result = orjson.loads(result_bytes)

    assert type(result) is dict
    assert len(result["results"]) == 3


def test_llama_server_lora(model_path):
    """Test loading and using a LoRA adapter.

    This test uses the stories15M_MOE model with a Shakespeare LoRA adapter.
    - Without LoRA (scale=0), the model generates bedtime story style text.
    - With LoRA (scale=1), the model generates Shakespearean style text.

    Based on: thirdparty/llama.cpp/tools/server/tests/unit/test_lora.py
    """
    params = xlc.CommonParams()

    params.model.path = os.path.join(model_path, "stories15M_MOE-F16.gguf")
    params.warmup = False
    params.n_predict = 64
    params.n_ctx = 256
    params.n_parallel = 1
    params.cpuparams.n_threads = 2
    params.cpuparams_batch.n_threads = 2
    params.sampling.seed = 42
    params.sampling.temp = 0.0
    params.sampling.top_k = 1

    # Create a CommonAdapterLoraInfo and set it via lora_adapters
    lora_path = os.path.join(model_path, "moe_shakespeare15M.gguf")
    lora_info = xlc.CommonAdapterLoraInfo(lora_path, 1.0)
    params.lora_adapters = [lora_info]

    # Verify the lora adapter was added and can be accessed via wrapper
    assert len(params.lora_adapters) == 1
    assert params.lora_adapters[0].path == lora_path
    assert params.lora_adapters[0].scale == 1.0

    # Test modifying the scale via the wrapper
    params.lora_adapters[0].scale = 0.5
    assert lora_info.scale == 0.5
    params.lora_adapters[0].scale = 1.0  # Reset back

    server = xlc.Server(params)

    # Test completion with LoRA applied
    complete_prompt = {
        "max_tokens": 64,
        "prompt": "Look in thy glass",
        "seed": 42,
        "temperature": 0.0,
    }

    result = server.handle_completions(complete_prompt)

    assert isinstance(result, dict)
    assert "code" not in result
    assert "choices" in result
    content = result["choices"][0]["text"]
    print(f"LoRA completion result: {content}")

    # With Shakespeare LoRA, expect Shakespearean-style words
    assert len(content) > 0
