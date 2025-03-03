# set path so `llama-cli` etc.. be in path
export PATH := $(PWD)/bin:$(PATH)

# models
MODEL := models/Llama-3.2-1B-Instruct-Q8_0.gguf

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
		wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q8_0.gguf

download: $(MODEL)
	@echo "minimal model downloaded to models directory"
