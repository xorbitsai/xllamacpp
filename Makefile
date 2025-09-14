# set path so `llama-cli` etc.. be in path
export PATH := $(PWD)/bin:$(PATH)
export MACOSX_DEPLOYMENT_TARGET := 11

# models
MODEL := bge-reranker-v2-m3-Q2_K.gguf

THIRDPARTY := $(PWD)/thirdparty
LLAMACPP := $(THIRDPARTY)/llama.cpp

.PHONY: all build wheel clean test download

all: build

build:
	@bash scripts/setup.sh
	python setup.py build_ext --inplace

wheel:
	@python setup.py bdist_wheel

clean:
	@rm -rf build dist src/llama.cpp src/*.egg-inf thirdparty/llama.cpp/build o .pytest_cache .coverage

test: build
	@pytest

$(MODEL):
	@mkdir -p models && cd models && \
		curl --output Llama-3.2-1B-Instruct-Q8_0.gguf -L https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf && \
		curl --output tinygemma3-Q8_0.gguf -L https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/tinygemma3-Q8_0.gguf && \
		curl --output mmproj-tinygemma3.gguf -L https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/mmproj-tinygemma3.gguf && \
		curl --output Qwen3-Embedding-0.6B-Q8_0.gguf -L https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf && \
		curl --output bge-reranker-v2-m3-Q2_K.gguf -L https://modelscope.cn/models/gpustack/bge-reranker-v2-m3-GGUF/resolve/master/bge-reranker-v2-m3-Q2_K.gguf

download: $(MODEL)
	@echo "minimal model downloaded to models directory"
